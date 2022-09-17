from dataclasses import dataclass
import numpy as np
from functools import cache
from itertools import takewhile
from typing import ClassVar
from argparse import Namespace

from gomoku.sequence import BASE_SCORE, Sequence, Coord, Block


def slice_up(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[max(y - size + 1, 0) : y + 1, x][::-1])


def slice_down(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[y : y + size, x])


def slice_left(board: np.ndarray, y: int, x: int, size: int) -> tuple[int]:
    return tuple(board[y, max(x - size + 1, 0) : x + 1][::-1])


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
    Coord(0, -1): slice_left,
    Coord(0, 1): slice_right,
    Coord(-1, 0): slice_up,
    Coord(1, 0): slice_down,
    Coord(-1, -1): slice_up_left,
    Coord(-1, 1): slice_up_right,
    Coord(1, -1): slice_down_left,
    Coord(1, 1): slice_down_right,
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

CAPTURE_MOVE_CASES = ((-1, 1, 1, -1), (1, -1, -1, 1))

NEXT_TURN_BONUS = BASE_SCORE // 3


def get_cell_values(size: int) -> np.ndarray:
    array = np.zeros((size, size), dtype=int)
    for y in range(size // 2 + 1):
        for x in range(size // 2 + 1):
            array[y, x] = array[y, -x - 1] = array[-y - 1, x] = array[
                -y - 1, -x - 1
            ] = (min(x, y) + 1)
    return array


@dataclass(init=False, repr=True, slots=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Coord, set[int]]
    seq_list: dict[int, Sequence]
    last_seq_id: int
    stones: set[Coord]
    capture: dict[int, int] | None
    last_chance: bool
    playing: int
    successors: set[Coord]

    # Class constants shared by all instances
    cell_values: ClassVar[np.ndarray]
    size: ClassVar[int]
    sequence_win: ClassVar[int]
    capture_win: ClassVar[int]
    free_double: ClassVar[bool]
    gravity: ClassVar[bool]
    debug: ClassVar[bool]

    def __init__(
        self,
        args: Namespace,
    ) -> None:
        self.cells = np.zeros((args.board, args.board), dtype=int)
        self.seq_map = {}
        self.seq_list = {}
        self.stones = set()
        self.successors = set()
        self.last_seq_id = 0
        self.capture = {1: 0, -1: 0} if args.capture_win else None
        self.last_chance = False
        self.playing = 1

        Board.cell_values = get_cell_values(args.board)
        Board.size = args.board
        Board.free_double = args.free_double
        Board.sequence_win = args.sequence_win
        Board.capture_win = args.capture_win
        Board.gravity = args.gravity
        Board.debug = args.debug

        Sequence.board_size = args.board
        Sequence.sequence_win = args.sequence_win
        Sequence.capture_win = args.capture_win

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "X", -1: "O"}
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
        s += f"Static Evaluation: {self.score} "
        s += f"(Sequences: {self.sequences_score}, Stones: {self.stones_score}, "
        s += f"Capture: {self.capture_score})\n"
        s += f"Last sequence id: {self.last_seq_id}\n"
        return s
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return (
            self.cells == other.cells
            and self.capture == other.capture
            and self.playing == other.playing
        )
    
    def __hash__(self) -> int:
        return hash((self.cells, self.capture, self.playing))

    @property
    def capture_score(self) -> int:
        """
        Score of capture
        """
        if self.capture is None:
            return 0
        return Sequence.capture_score(tuple(self.capture.items()))

    @property
    def stones_score(self) -> int:
        """
        Score of all stones on the board
        """
        return sum(self.cell_values[coord] * self.cells[coord] for coord in self.stones)

    @property
    def sequences_score(self) -> int:
        """
        Score of all sequences on the board
        """
        score = {1: 0, -1: 0}
        best = 0
        for seq in self.seq_list.values():
            current = seq.score(self.capture) * seq.player
            if seq.player == -self.playing and current > best:
                best = current
            else:
                score[seq.player] += current * seq.player
        return sum(score.values()) + (best * NEXT_TURN_BONUS * -self.playing)

    @property
    def score(self) -> int:
        """
        Static evaluation of the board
        """
        return self.sequences_score + self.stones_score + self.capture_score

    def is_free_double(self, pos: Coord, player: int) -> bool:
        """
        Check if a move introduces a double free three
        """
        if not self.free_double:
            return False
        free_double = 0
        for id in self.seq_map.get(pos, set()):
            seq = self.seq_list[id]
            if seq.player != player:
                continue
            if (
                len(seq) == Board.sequence_win - 3
                and seq.is_blocked == Block.NO
                and pos in seq.cost_cells
            ):
                if seq.start < pos < seq.end:
                    free_double += 1
                else:
                    p = pos - seq.dir if pos < seq.start else pos + seq.dir
                    if p.in_range(Board.size) and self.cells[p] != -player:
                        free_double += 1
        return free_double >= 2

    def is_game_over(self) -> int:
        """
        Check if the game is over.
        """
        if self.capture and any(
            c >= Sequence.capture_win for c in self.capture.values()
        ):
            return 1 if self.capture[1] >= Sequence.capture_win else 2
        for seq in self.seq_list.values():
            if seq.is_win:
                if not self.capture or self.last_chance:
                    return seq.player if seq.player == 1 else 2
                if self.capturable_stones_in_sequences(seq):
                    self.last_chance = True
                    return 0
                return seq.player if seq.player == 1 else 2
        return 0 if 0 in self.cells else -1

    def get_valid_pos(self, y: int, x: int) -> Coord | None:
        """
        Get the position of the stone depending on the gravity option
        """
        if not Board.gravity:
            pos = Coord(y, x)
            return pos if self.can_place(pos) else None
        offset = np.argmax(self.cells[::-1, x] == 0)
        if self.cells[Board.size - offset - 1, x] == 0:
            return Coord(Board.size - offset - 1, x)

    def can_place(self, pos: Coord) -> bool:
        """
        Returns whether the given position is a valid move,
        meaning not out of bounds and not already occupied.
        """
        return pos.in_range(Board.size) and self.cells[pos] == 0

    def remove_sequence(self, id: int) -> None:
        """
        Remove a sequence from the board.
        """
        self.seq_list.pop(id)
        for cell in self.seq_map:
            if id in self.seq_map[cell]:
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
            self.seq_map.get(cell, set()).discard(id)
        if from_list:
            seq.spaces = (
                (start, seq.spaces[1]) if index == 0 else (seq.spaces[0], start)
            )

    def add_sequence(self, seq: Sequence) -> None:
        """
        Add a new sequence to the sequence map and list.
        """
        if seq.is_dead:
            return
        if seq.id == -1:
            seq.id = self.last_seq_id
            self.last_seq_id += 1
        for cell in seq:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.growth_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.block_cells:
            if cell.in_range(Board.size):
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
            if not flank.in_range(Board.size) or self.cells[flank] == -seq.player:
                self.add_block_to_sequence(flank, id, Block.HEAD)
            else:
                self.remove_sequence_spaces(pos - seq.dir, id, Block.HEAD)
            seq.extend_head(pos, self.get_spaces(pos, -seq.dir, -seq.player))
            self.add_sequence_spaces(id)
        elif pos > seq.end:
            self.remove_sequence_spaces(seq.start - seq.dir, id, Block.HEAD)
            flank = pos + seq.dir
            if not flank.in_range(Board.size) or self.cells[flank] == -seq.player:
                self.add_block_to_sequence(flank, id, Block.TAIL)
            else:
                self.remove_sequence_spaces(pos + seq.dir, id, Block.TAIL)
            seq.extend_tail(pos, self.get_spaces(pos, seq.dir, -seq.player))
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
        if length < Board.sequence_win:
            return self.remove_sequence(id)
        self.remove_sequence_spaces(pos, id, block, from_list=True)
        if to_list:
            seq.is_blocked = Block(block + seq.is_blocked)
        if pos.in_range(Board.size):
            self.seq_map.setdefault(pos, set()).add(seq.id)

    def split_sequence_at_block(self, pos: Coord, id: int) -> None:
        """
        Split a sequence into two sequences after an opponent block in the sequence.
        """
        seq = self.seq_list[id]
        head, tail = seq.split_at_blocked_hole(pos)
        self.remove_sequence(id)
        self.add_sequence(head)
        self.add_sequence(tail)

    def update_sequences_at_added_stone(self, pos: Coord, player: int) -> None:
        """
        Update sequences at the position of a newly added stone.
        """
        visited = set()
        for id in self.seq_map.get(pos, set()).copy():
            seq = self.seq_list[id]
            visited.update((seq.dir, -seq.dir))
            if seq.player == player and not pos in seq:
                if not seq.can_extend(pos):
                    continue
                self.extend_sequence(pos, id)
            elif pos in seq.holes:
                self.split_sequence_at_block(pos, id)
            elif not pos in seq.flank_cells:
                block = Block.HEAD if pos < seq.start else Block.TAIL
                self.remove_sequence_spaces(pos, id, block, from_list=True)
                if seq.capacity < Board.sequence_win:
                    self.remove_sequence(id)
            else:
                block = seq.is_block(pos)
                self.add_block_to_sequence(pos, id, block, to_list=True)
        for d in visited.intersection(DIRECTIONS).symmetric_difference(DIRECTIONS):
            self.add_sequence(self.get_sequence(pos, d, player))

    def add_move(self, pos: Coord, player: int) -> list[Coord]:
        """
        Adds a move to the board by placing the player's stone id at the given position
        and updating the sequences around the new stone.
        """
        capturable = []
        self.cells[pos] = player
        self.stones.add(pos)
        if self.capture:
            capturable = self.capturable_stones(pos, CAPTURE_MOVE_CASES)
            for stone in capturable:
                self.cells[stone] = 0
                self.stones.remove(stone)
            self.capture[player] += len(capturable) // 2
            for stone in capturable:
                self.update_sequences_at_removed_stone(stone)
        self.update_sequences_at_added_stone(pos, player)
        if len(capturable) > 0:
            self.successors = self.get_successors()
        else:
            self.update_successors(pos)
        if Board.debug:
            print(self)
        self.playing = -player
        return capturable

    def reduce_sequence_head(self, id: int) -> None:
        """
        Remove the start of a sequence.
        """
        tmp_seq = self.seq_list[id].copy()
        tmp_seq.reduce_head()
        if tmp_seq.is_dead:
            return self.remove_sequence(id)
        self.seq_list[id] = tmp_seq
        seq = self.seq_list[id]
        self.remove_sequence_spaces(seq.start - seq.dir, id, Block.HEAD)
        seq.spaces = min(seq.spaces[0], Board.sequence_win - 1), seq.spaces[1]
        self.add_sequence_spaces(id)

    def reduce_sequence_tail(self, id: int) -> None:
        """
        Remove the end of a sequence.
        """
        tmp_seq = self.seq_list[id].copy()
        tmp_seq.reduce_tail()
        if tmp_seq.is_dead:
            return self.remove_sequence(id)
        self.seq_list[id] = tmp_seq
        seq = self.seq_list[id]
        self.remove_sequence_spaces(seq.end + seq.dir, id, Block.TAIL)
        seq.spaces = seq.spaces[0], min(seq.spaces[1], Board.sequence_win - 1)
        self.add_sequence_spaces(id)

    def split_sequence_at_removed_stone(self, id: int) -> None:
        """
        Split a sequence into two sequences after a removed stone in the sequence.
        """
        seq = self.seq_list[id]
        stones = filter(lambda x: self.cells[x] == seq.player, seq.rest_cells)
        seq_list = set(self.get_sequence(s, seq.dir, seq.player) for s in stones)
        self.remove_sequence(id)
        for new_seq in seq_list:
            self.add_sequence(new_seq)

    def replace_sequence(self, id: int) -> None:
        """
        Replace a sequence with a new sequence.
        """
        seq = self.seq_list[id]
        new = self.get_sequence(seq.start, seq.dir, seq.player)
        new.id = id
        self.remove_sequence(id)
        self.add_sequence(new)

    def find_and_replace_sequences(self, pos: Coord, dir: Coord) -> None:
        """
        Find a stone in the given direction and replace the sequences at the stone
        """
        board_slice = SLICE_MAP[dir](self.cells, pos.y, pos.x, Board.sequence_win - 1)
        dist = self.find_stone(board_slice)
        if dist is None:
            return
        new_pos = pos + dir * dist
        for id in self.seq_map.get(new_pos, set()).copy():
            seq = self.seq_list[id]
            if seq.dir == dir or seq.dir == -dir:
                return self.replace_sequence(id)
        self.add_sequence(self.get_sequence(new_pos, dir, self.cells[new_pos]))

    def update_sequences_at_removed_stone(self, pos: Coord) -> None:
        """
        Updates the sequences affected by the removal of a stone at the given position.
        """
        visited = set()
        for id in self.seq_map.get(pos, set()).copy():
            seq = self.seq_list[id]
            if pos == seq.start:
                self.reduce_sequence_head(id)
                visited.add(seq.dir)
            elif pos == seq.end:
                self.reduce_sequence_tail(id)
                visited.add(-seq.dir)
            elif pos in seq.block_cells:
                self.replace_sequence(id)
                visited.add(seq.dir if pos < seq.start else -seq.dir)
            else:
                self.split_sequence_at_removed_stone(id)
                visited.update((seq.dir, -seq.dir))
        for d in visited.symmetric_difference(SLICE_MAP.keys()):
            self.find_and_replace_sequences(pos, d)

    def capturable_stones(self, pos: Coord, cases: tuple[tuple[int]]) -> list[Coord]:
        """
        Returns a list of capturable stones from the given position.
        """
        capturable = []
        for dir in SLICE_MAP.keys():
            board_slice = SLICE_MAP[dir](
                self.cells, pos.y, pos.x, Board.sequence_win - 1
            )
            if self.capturable_slice(board_slice, cases):
                capturable.extend([pos + dir, pos + dir + dir])
        return capturable

    def get_successors_around_stone(self, pos: Coord) -> list[Coord]:
        """
        Returns a list of positions around the given position.
        """
        return [pos + off for off in NEIGHBORS_OFFSET if self.can_place(pos + off)]

    def get_successors(self) -> set[Coord]:
        """
        Returns the coordinates of the neighbor cells of all stones in a 2-cell radius
        """
        successors = set()
        for stone in self.stones:
            successors.update(self.get_successors_around_stone(stone))
        return successors

    def update_successors(self, pos: Coord) -> None:
        """
        Update the successors of the board.
        """
        self.successors.update(self.get_successors_around_stone(pos))
        self.successors.discard(pos)

    @staticmethod
    @cache
    def capturable_slice(slice: np.ndarray, cases: tuple[tuple[int]]) -> bool:
        """
        Returns True if the given slice contains a capturable sequence.
        """
        return any(map(lambda case: np.array_equal(slice, np.array(case)), cases))

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
                if holed or cell == -player:
                    return tuple(shape)
                holed = True
        return tuple(shape) if sub_len == 0 else tuple(shape) + (sub_len,)

    @staticmethod
    @cache
    def find_stone(slice: np.ndarray) -> int | None:
        """
        Returns the position of the first stone in the given slice.
        """
        for i, cell in enumerate(slice):
            if cell != 0:
                return i
        return None

    def get_spaces(self, pos: Coord, dir: Coord, opponent: int) -> int:
        """
        Returns the number of empty or ally cells after a sequence in a direction.
        """
        in_range = lambda p: p.in_range(Board.size) and self.cells[p] != opponent
        return (
            sum(1 for _ in takewhile(in_range, pos.range(dir, Board.sequence_win))) - 1
        )

    def get_half_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        """
        Returns a sequence in a direction from a position.
        """
        board_slice = SLICE_MAP[dir](self.cells, pos.y, pos.x, Board.sequence_win)
        shape = self.slice_to_shape(board_slice)
        seq = Sequence(player, shape, pos, dir)
        seq.spaces = self.get_spaces(seq.end, seq.dir, -seq.player)
        seq.is_blocked = Block.NO if seq.spaces != 0 else seq.dir.to_block()
        if seq.dir.to_block() == Block.HEAD:
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
        return head + tail if dir < -dir else tail + head

    def capturable_stones_in_sequences(self, seq: Sequence) -> bool:
        """
        Returns whether a sequence can be broken/reduced with a capture
        """
        for stone in seq:
            visited = set()
            for id in self.seq_map.get(stone, set()).copy():
                s = self.seq_list[id]
                if seq.id == id or seq.player != s.player:
                    continue
                if s.capturable_sequence() != 0:
                    return True
                visited.update((s.dir, -s.dir))
            for d in visited.symmetric_difference(SLICE_MAP.keys()):
                s = self.get_sequence(stone, d, seq.player)
                if s.capturable_sequence() != 0:
                    return True
        return False
