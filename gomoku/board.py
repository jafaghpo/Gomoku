from dataclasses import dataclass
import numpy as np
from functools import cache
from sys import exit
from itertools import takewhile

from gomoku.sequence import Sequence, Coord, Block, MAX_SEQ_LEN


def slice_up(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[max(y - size - 1, 0) : y + 1, x][::-1])


def slice_down(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[y : y + size, x])


def slice_left(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[y, max(x - size - 1, 0) : x + 1][::-1])


def slice_right(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[y, x : x + size])


def slice_up_left(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    y, x = board.shape[0] - 1 - y, board.shape[1] - 1 - x
    return tuple(np.fliplr(np.flipud(board))[y:, x:].diagonal()[:size])


def slice_up_right(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    y = board.shape[0] - 1 - y
    return tuple(np.flipud(board)[y:, x:].diagonal()[:size])


def slice_down_left(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    x = board.shape[1] - 1 - x
    return tuple(np.fliplr(board)[y:, x:].diagonal()[:size])


def slice_down_right(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(np.diagonal(board[y:, x:])[:size])


SLICE_MAP = {
    (0, -1): slice_left,
    (0, 1): slice_right,
    (-1, 0): slice_up,
    (1, 0): slice_down,
    (-1, -1): slice_up_left,
    (-1, 1): slice_up_right,
    (1, -1): slice_down_left,
    (1, 1): slice_down_right,
}

NEIGHBORS_OFFSET = tuple(
    (
        Coord(-2, -2),
        Coord(-2, 0),
        Coord(-2, 2),
        Coord(-1, -1),
        Coord(-1, 0),
        Coord(-1, 1),
        Coord(0, -2),
        Coord(0, -1),
        Coord(0, 1),
        Coord(0, 2),
        Coord(1, -1),
        Coord(1, 0),
        Coord(1, 1),
        Coord(2, -2),
        Coord(2, 0),
        Coord(2, 2),
    )
)

DIRECTIONS = tuple(map(Coord._make, ((0, -1), (-1, -1), (-1, 0), (-1, 1))))


@dataclass(init=False, repr=True, slots=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Coord, set[int]]
    seq_list: dict[int, Sequence]
    last_seq_id: int
    last_move: Coord | None
    stones: set[Coord]

    def __init__(self, shape: tuple[int, int] = (19, 19)) -> None:
        self.cells = np.zeros(shape, dtype=int)
        self.seq_map = {}
        self.seq_list = {}
        self.stones = set()
        self.last_move = None
        self.last_seq_id = 0
        Sequence.bounds = Coord(*shape)

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "X", 2: "O"}
        s = "Cells:\n"
        s += "\n".join(
            " ".join(map(lambda cell: player_repr[cell], row)) for row in self.cells
        )
        s += "\nStones: " + " ".join(str(stone) for stone in self.stones) + "\n"
        for seq in self.seq_list.values():
            s += str(seq)
        s += "Sequence map:\n"
        for cell, seq_ids in self.seq_map.items():
            if seq_ids:
                s += f"{cell}: {seq_ids}\n"
        s += "Last sequence id: " + str(self.last_seq_id) + "\n"
        s += "Last move: " + str(self.last_move) + "\n"
        return s

    def get_pos_c4(self, x: int):
        for y in range(5, -1, -1):
            if self.cells[y, x] == 0:
                return Coord(y, x)

    def can_place_c4(self, x: int) -> bool:
        print(self.cells[(x, 0)])
        return self.cells[(x, 0)] == 0

    def can_place(self, pos: Coord) -> bool:
        """
        Returns whether the given position is a valid move,
        meaning not out of bounds and not already occupied.
        """
        return pos.in_range(self.cells.shape) and self.cells[pos] == 0

    def remove_sequence(self, id: int) -> None:
        """
        Remove a sequence from the board.
        """
        seq = self.seq_list.pop(id)
        for cell in seq.rest_cells:
            self.seq_map[cell].discard(id)
        for cell in seq.growth_cells:
            self.seq_map[cell].discard(id)
        for cell in seq.block_cells:
            if cell.in_range(self.cells.shape):
                self.seq_map[cell].discard(id)

    def remove_sequence_spaces(
        self, pos: Coord, id: int, block: Block, from_list: bool = False
    ) -> None:
        """
        Remove the sequence spaces from sequence map.
        """
        seq = self.seq_list[id]
        index = int(block) - 1
        spaces = seq.space_cells[index]
        try:
            start = spaces.index(pos)
        except ValueError:
            return
        for cell in spaces[start:]:
            self.seq_map[cell].discard(id)
        if from_list:
            seq.spaces = (
                (start, seq.spaces[1]) if index == 0 else (seq.spaces[0], start)
            )

    def add_sequence(self, seq: Sequence) -> None:
        """
        Add a sequence to the board.
        """
        if seq.is_dead:
            return
        if seq.id == -1:
            seq.id = self.last_seq_id
            self.last_seq_id += 1
        for cell in seq.rest_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.growth_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.block_cells:
            if cell.in_range(self.cells.shape):
                self.seq_map.setdefault(cell, set()).add(seq.id)
        self.seq_list[seq.id] = seq

    def add_sequence_spaces(self, id: int) -> None:
        """
        Add the sequence spaces to sequence map.
        """
        for side in self.seq_list[id].space_cells:
            for cell in side:
                self.seq_map.setdefault(cell, set()).add(id)

    def extend_sequence(self, pos: Coord, id: int) -> None:
        """
        Extend a sequence with the given position.
        """
        seq = self.seq_list[id]
        if pos < seq.start:
            self.remove_sequence_spaces(seq.end + seq.dir, id, Block.TAIL)
            flank = pos - seq.dir
            if (
                not flank.in_range(self.cells.shape)
                or self.cells[flank] == seq.player ^ 3
            ):
                self.add_block_to_sequence(flank, id, Block.HEAD)
            else:
                self.remove_sequence_spaces(pos - seq.dir, id, Block.HEAD)
            seq.extend_head(pos, self.get_spaces(pos, -seq.dir, seq.player ^ 3))
            self.add_sequence_spaces(id)
        elif pos > seq.end:
            self.remove_sequence_spaces(seq.start - seq.dir, id, Block.HEAD)
            flank = pos + seq.dir
            if (
                not flank.in_range(self.cells.shape)
                or self.cells[flank] == seq.player ^ 3
            ):
                self.add_block_to_sequence(flank, id, Block.TAIL)
            else:
                self.remove_sequence_spaces(pos + seq.dir, id, Block.TAIL)
            seq.extend_tail(pos, self.get_spaces(pos, seq.dir, seq.player ^ 3))
            self.add_sequence_spaces(id)
        else:
            seq.extend_hole(pos)

    def add_block_to_sequence(
        self, pos: Coord, id: int, block: Block, to_list: bool = False
    ) -> None:
        """
        Add a block to a sequence and remove the sequence if it is dead.
        """
        seq = self.seq_list[id]
        space = seq.spaces[1] if block == Block.HEAD else seq.spaces[0]
        length = space + pos.distance(seq.end if block == Block.HEAD else seq.start)
        if length < MAX_SEQ_LEN:
            return self.remove_sequence(id)
        self.remove_sequence_spaces(pos, id, block, from_list=True)
        if to_list:
            seq.is_blocked = Block(block + seq.is_blocked)
        if pos.in_range(self.cells.shape):
            self.seq_map.setdefault(pos, set()).add(seq.id)

    def split_sequence(self, pos: Coord, id: int) -> None:
        """
        Split a sequence into two sequences.
        """
        seq = self.seq_list[id]
        self.remove_sequence(id)
        head = self.get_sequence(seq.start, seq.dir, seq.player)
        tail = self.get_sequence(seq.end, seq.dir, seq.player)
        if not head.is_dead:
            self.add_sequence(head)
        if not tail.is_dead:
            self.add_sequence(tail)

    def update_sequences(self, pos: Coord, player: int) -> None:
        """
        Update the sequences that are affected by the given position,
        removing sequences that are no longer valid and adding new ones.
        """
        visited = set()
        for id in self.seq_map.get(pos, set()).copy():
            seq = self.seq_list[id]
            visited.update((seq.dir, -seq.dir))
            if seq.player == player:
                if not seq.can_extend(pos):
                    continue
                self.extend_sequence(pos, id)
            elif pos in seq.holes:
                self.split_sequence(pos, id)
            elif not pos in seq.cost_cells:
                block = Block.HEAD if pos < seq.start else Block.TAIL
                self.remove_sequence_spaces(pos, id, block, from_list=True)
                if seq.capacity < MAX_SEQ_LEN:
                    self.remove_sequence(id)
            else:
                block = seq.can_pos_block(pos)
                self.add_block_to_sequence(pos, id, block, to_list=True)
        for d in visited.intersection(DIRECTIONS).symmetric_difference(DIRECTIONS):
            self.add_sequence(self.get_sequence(pos, d, player))

    def add_move(self, pos: Coord, player: int) -> None:
        """
        Adds a move to the board by placing the player's stone id at the given position
        and updating the sequences around the new stone.
        """
        self.cells[pos] = player
        self.stones.add(pos)
        self.last_move = pos
        self.update_sequences(pos, player)
        print(self)

    def get_neighbors(self) -> set[Coord]:
        """
        Returns the coordinates of the neighbor cells of all stones in a 2-cell radius
        """
        children = set()
        for stone in self.stones:
            raw_neighbors = map(lambda offset: stone + offset, NEIGHBORS_OFFSET)
            neighbors = list(filter(lambda n: self.can_place(n), raw_neighbors))
            children.update(neighbors)
        return children

    @staticmethod
    @cache
    def slice_to_shape(slice: np.ndarray) -> tuple[int]:
        """
        Returns the shape of a slice of the board.

        A shape is a tuple containing the length of all subsequences separated by empty
        cells in a direction.
        ex: [1, 1, 0, 1, 2, 0] -> (2, 1)
        We stop the sequence at index 4 because '2' is the opponent stone id.
        """
        player = slice[0]
        shape = []
        sub_len = 1
        holed = False
        for cell in slice[1:]:
            if cell == player:
                sub_len += 1
                holed = False
            else:
                if sub_len > 0:
                    shape.append(sub_len)
                    sub_len = 0
                if holed or cell == player ^ 3:
                    return tuple(shape)
                holed = True
        return tuple(shape) if sub_len == 0 else tuple(shape) + (sub_len,)

    def get_spaces(self, pos: Coord, dir: Coord, opponent: int) -> int:
        """
        Returns the number of empty or ally cells after a sequence in a direction.
        """
        in_range = lambda p: p.in_range(self.cells.shape) and self.cells[p] != opponent
        return sum(1 for _ in takewhile(in_range, pos.range(dir, MAX_SEQ_LEN))) - 1

    def get_half_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        """
        Returns a sequence in a direction from a position.
        """
        board_slice = SLICE_MAP[dir](self.cells, pos.y, pos.x, MAX_SEQ_LEN)
        shape = self.slice_to_shape(board_slice)
        seq = Sequence(player, shape, pos, dir)
        seq.spaces = self.get_spaces(seq.end, seq.dir, seq.player ^ 3)
        seq.is_blocked = Block.NO if seq.spaces != 0 else seq.dir.get_block_dir()
        if seq.dir.get_block_dir() == Block.HEAD:
            seq.start = seq.end
            seq.shape = seq.shape[::-1]
            seq.dir = -seq.dir
        return seq

    def get_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        """
        Returns a sequence that combines the sequences starting from the same cell
        but having opposite directions.
        """
        head = self.get_half_sequence(pos, dir, player)
        tail = self.get_half_sequence(pos, -dir, player)
        return head + tail
