from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from functools import cache
from typing import ClassVar
from argparse import Namespace

from gomoku.sequence import Sequence, Block, BASE_SCORE
import gomoku.coord as coord
from gomoku.coord import Coord, in_bound


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
    (0, -1): slice_left,
    (0, 1): slice_right,
    (-1, 0): slice_up,
    (1, 0): slice_down,
    (-1, -1): slice_up_left,
    (-1, 1): slice_up_right,
    (1, -1): slice_down_left,
    (1, 1): slice_down_right,
}

NEIGHBORS = (
    (-1, -1),
    (-2, -2),
    (-1, 0),
    (-2, 0),
    (-1, 1),
    (-2, 2),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 2),
    (1, 0),
    (2, 0),
    (1, -1),
    (2, -2),
    (0, -1),
    (0, -2),
)

CLOSE_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

DIRECTIONS = ((0, -1), (-1, -1), (-1, 0), (-1, 1))

CAPTURE_MOVE_CASES = ((-1, 1, 1, -1), (1, -1, -1, 1))

NEXT_TURN_BONUS = BASE_SCORE


class GameOver(IntEnum):
    NONE = 0
    DRAW = 1
    BLACK_SEQUENCE_WIN = 2
    BLACK_CAPTURE_WIN = 3
    WHITE_SEQUENCE_WIN = 4
    WHITE_CAPTURE_WIN = 5


def get_cell_values(size: int) -> np.ndarray:
    array = np.zeros((size, size), dtype=int)
    for y in range(size // 2 + 1):
        for x in range(size // 2 + 1):
            array[y, x] = array[y, -x - 1] = array[-y - 1, x] = array[
                -y - 1, -x - 1
            ] = (min(x, y) + 1)
    return array


def get_cell_neighbors(size: int, offsets: tuple[Coord]) -> dict[Coord, tuple[Coord]]:
    neighbors = {}
    for y in range(size):
        for x in range(size):
            lst = []
            for offset in offsets:
                neighbor = (y + offset[0], x + offset[1])
                if in_bound(neighbor, size):
                    lst.append(neighbor)
            neighbors[(y, x)] = tuple(lst)
    return neighbors


@dataclass(init=False, repr=True, slots=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Coord, set[int]]
    seq_list: dict[int, Sequence]
    stones: list[Coord]
    capture: dict[int, int] | None
    last_chance: bool
    playing: int
    successors: list[Coord]
    move_history: list[Coord, list[Coord]]

    # Class constants shared by all instances
    cell_values: ClassVar[np.ndarray]
    cell_neighbors = ClassVar[dict[Coord, tuple[Coord]]]
    cell_close_neighbors = ClassVar[dict[Coord, tuple[Coord]]]
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
        self.stones = []
        self.successors = set()
        self.capture = {1: 0, -1: 0} if args.capture_win else None
        self.last_chance = False
        self.playing = 1
        self.move_history = []

        Board.cell_values = get_cell_values(args.board)
        Board.cell_neighbors = get_cell_neighbors(args.board, NEIGHBORS)
        Board.cell_close_neighbors = get_cell_neighbors(args.board, CLOSE_NEIGHBORS)
        Board.size = args.board
        Board.free_double = args.free_double
        Board.sequence_win = args.sequence_win
        Board.capture_win = args.capture_win
        Board.gravity = args.gravity
        Board.debug = args.debug

        Sequence.board_size = args.board
        Sequence.sequence_win = args.sequence_win
        Sequence.capture_win = args.capture_win
        Sequence.capture_weight = 1

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "\u001b[32mX\033[00m", -1: "\u001b[36mO\033[00m"}
        cells = [[player_repr[col] for col in row] for row in self.cells]
        y, x = self.last_move
        cells[y][x] = f"\033[91m{'X' if self.cells[y][x] == 1 else 'O'}\033[00m"
        for y, x in self.successors:
            cells[y][x] = f"\033[33m*\033[00m"
        s = "\nBoard:\n"
        s += " ".join([str(i % 10) for i in range(self.size)]) + "\n"
        s += "\n".join(" ".join(row + [str(i)]) for i, row in enumerate(cells))
        s += "\nSequence list:\n"
        for seq in sorted(self.seq_list.values()):
            s += f"  {seq.to_string(self.playing, self.capture)}\n"
        s += f"Evaluation: {self.score} "
        s += f"(Sequences: {self.sequences_score}, Stones: {self.stones_score}, "
        s += f"Capture: {self.capture_score})\n"
        s += f"Playing: {self.playing}\n"
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
        return hash(
            self.cells.tobytes()
            + str(self.capture).encode()
            + str(self.playing).encode()
        )

    def reset(self) -> None:
        self.cells = np.zeros((Board.size, Board.size), dtype=int)
        self.seq_map = {}
        self.seq_list = {}
        self.stones = []
        self.successors = set()
        self.capture = {1: 0, -1: 0} if Board.capture_win else None
        self.last_chance = False
        self.playing = 1
        self.move_history = []
    
    @property
    def last_move(self) -> Coord:
        return self.move_history[-1][0]

    @property
    def capture_score(self) -> int:
        """
        Score of capture
        """
        if self.capture is None:
            return 0
        black_score = Sequence.capture_score(self.capture[1]) * 2
        white_score = Sequence.capture_score(self.capture[-1]) * -1 * 2
        return black_score + white_score

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
            current = seq.score(self.playing, self.capture) * seq.player
            if seq.player == self.playing and current > best:
                best = current
            else:
                score[seq.player] += current * seq.player
        return sum(score.values()) + (best * NEXT_TURN_BONUS * self.playing)

    @property
    def score(self) -> int:
        """
        Static evaluation of the board
        """
        return self.sequences_score + self.stones_score + self.capture_score

    @property
    def new_available_id(self) -> int:
        """
        Returns a new available id for a sequence
        """
        for id in range(100000):
            if id not in self.seq_list:
                return id

    def is_free_double(self, pos: Coord, player: int) -> bool:
        """
        Check if a move introduces a double free three
        """
        if not Board.free_double:
            return False
        free_double = 0
        self.cells[pos] = player
        capturable = self.capturable_stones(pos, CAPTURE_MOVE_CASES)
        if capturable:
            self.cells[pos] = 0
            return False
        for dir in DIRECTIONS:
            seq = self.get_sequence(pos, dir, player)
            if (
                len(seq) == min(Board.sequence_win - 2, 3)
                and seq.is_blocked == Block.NO
                and seq.nb_holes <= 1
            ):
                free_double += 1
        self.cells[pos] = 0
        return free_double >= 2

    def is_capture_win(self) -> bool:
        """
        Check if a capture win is possible
        """
        return self.capture and max(self.capture.values()) >= Board.capture_win

    def is_game_over(self) -> int:
        """
        Check if the game is over.
        """
        if self.is_capture_win():
            if self.capture[1] >= Board.capture_win:
                return GameOver.BLACK_CAPTURE_WIN
            else:
                return GameOver.WHITE_CAPTURE_WIN
        if len(self.stones) // 2 + 1 < Board.sequence_win:
            return GameOver.NONE
        for seq in self.seq_list.values():
            if seq.is_win:
                if self.capture:
                    count, _ = self.capturable_stones_in_sequences(seq)
                    if count > 0:
                        if self.last_chance:
                            if seq.player == 1:
                                return GameOver.BLACK_SEQUENCE_WIN
                            else:
                                return GameOver.WHITE_SEQUENCE_WIN
                        print(f"Player {1 if seq.player == 1 else 2} created ", end='')
                        print("a winning sequence but the opponent ", end='')
                        print("can still capture and break it")
                        self.last_chance = True
                        return GameOver.NONE
                if seq.player == 1:
                    return GameOver.BLACK_SEQUENCE_WIN
                else:
                    return GameOver.WHITE_SEQUENCE_WIN
        if self.last_chance:
            self.last_chance = False
        return GameOver.NONE if 0 in self.cells else GameOver.DRAW

    def get_valid_pos(self, y: int, x: int) -> Coord | None:
        """
        Get the position of the stone depending on the gravity option
        """
        if not Board.gravity:
            pos = (y, x)
            return pos if self.can_place(pos) else None
        offset = np.argmax(self.cells[::-1, x] == 0)
        if self.cells[Board.size - offset - 1, x] == 0:
            return (Board.size - offset - 1, x)

    def can_place(self, pos: Coord) -> bool:
        """
        Returns whether the given position is a valid move,
        meaning not out of bounds and not already occupied.
        """
        if coord.in_bound(pos, Board.size) and self.cells[pos] == 0:
            if self.gravity:
                below = (pos[0] + 1, pos[1])
                return below[0] >= Board.size or self.cells[below] != 0
            return True
        return False

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
        if block != Block.BOTH:
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
        else:
            for cell in seq.space_cells:
                for space in cell:
                    self.seq_map.get(space, set()).discard(id)
            if from_list:
                seq.spaces = (0, 0)

    def add_sequence(self, seq: Sequence) -> None:
        """
        Add a new sequence to the sequence map and list.
        """
        if seq.is_dead:
            return
        if seq.id == -1:
            seq.id = self.new_available_id
        for cell in seq:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.growth_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.block_cells:
            if coord.in_bound(cell, Board.size):
                self.seq_map.setdefault(cell, set()).add(seq.id)
        self.seq_list[seq.id] = seq

    def add_sequence_spaces(self, id: int, index: int = 2) -> None:
        """
        Add the sequence spaces to sequence map.
        """
        if index != 2:
            for cell in self.seq_list[id].space_cells[index]:
                self.seq_map.setdefault(cell, set()).add(id)
        else:
            for side in self.seq_list[id].space_cells:
                for cell in side:
                    self.seq_map.setdefault(cell, set()).add(id)

    def extend_sequence(self, pos: Coord, id: int) -> None:
        """
        Extend a sequence with the given position.
        """
        seq = self.seq_list[id]
        if pos < seq.start:
            self.extend_head_sequence(pos, seq)
        elif pos > seq.end:
            self.extend_tail_sequence(pos, seq)
        else:
            seq.extend_hole(pos)
        for cells in seq.space_cells:
            for cell in cells[:2]:
                if self.cells[cell] == seq.player:
                    self.extend_sequence(cell, id)

    def extend_head_sequence(self, pos: Coord, seq: Sequence) -> None:
        """
        Extend a sequence head with the given position.
        """
        self.remove_sequence_spaces(coord.sub(pos, seq.dir), seq.id, Block.HEAD)
        seq.extend_head(pos, self.get_spaces(pos, coord.neg(seq.dir), -seq.player))
        self.add_sequence_spaces(seq.id, index=0)
        if seq.spaces[0] == 0:
            self.add_block_to_sequence(coord.sub(pos, seq.dir), seq.id, Block.HEAD)

    def extend_tail_sequence(self, pos: Coord, seq: Sequence) -> None:
        """
        Extend a sequence tail with the given position.
        """
        self.remove_sequence_spaces(coord.add(pos, seq.dir), seq.id, Block.TAIL)
        seq.extend_tail(pos, self.get_spaces(pos, seq.dir, -seq.player))
        self.add_sequence_spaces(seq.id, index=1)
        if seq.spaces[1] == 0:
            self.add_block_to_sequence(coord.add(pos, seq.dir), seq.id, Block.TAIL)

    def add_block_to_sequence(
        self, pos: Coord, id: int, block: Block, to_list: bool = False
    ) -> None:
        """
        Add a block to a sequence and remove the sequence if it is dead.
        """
        seq = self.seq_list[id]
        space = seq.spaces[1] if block == Block.HEAD else seq.spaces[0]
        length = space + coord.distance(
            pos, seq.end if block == Block.HEAD else seq.start
        )
        if length < Board.sequence_win:
            return self.remove_sequence(id)
        self.remove_sequence_spaces(pos, id, block, from_list=True)
        if to_list:
            if block + seq.is_blocked > 3:
                seq.is_blocked = Block(block)
            else:
                seq.is_blocked = Block(block + seq.is_blocked)
        if coord.in_bound(pos, Board.size):
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
            visited.update((seq.dir, coord.neg(seq.dir)))
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

    def undo_last_move(self) -> None:
        """
        Undo the last move.
        """
        move, captures = self.move_history.pop()
        self.cells[move] = 0
        self.stones.remove(move)
        self.update_sequences_at_removed_stone(move)
        if captures:
            self.capture[-self.playing] -= len(captures) // 2
            for pos in captures:
                self.cells[pos] = self.playing
                self.stones.append(pos)
                self.update_sequences_at_added_stone(pos, self.playing)
        self.playing *= -1

    def add_move(self, pos: Coord, sort_successors: bool = False) -> list[Coord]:
        """
        Adds a move to the board by placing the player's stone id at the given position
        and updating the sequences around the new stone.
        """
        capturable = []
        self.cells[pos] = self.playing
        self.stones.append(pos)
        if self.capture:
            capturable = self.capturable_stones(pos, CAPTURE_MOVE_CASES)
            for stone in capturable:
                self.cells[stone] = 0
                self.stones.remove(stone)
            self.capture[self.playing] += len(capturable) // 2
            for stone in capturable:
                self.update_sequences_at_removed_stone(stone)
        self.update_sequences_at_added_stone(pos, self.playing)
        self.playing = -self.cells[pos]
        self.successors = self.get_successors()
        if sort_successors:
            lst = []
            for cell in self.successors:
                self.add_move(cell)
                lst.append((self.score, cell))
                self.undo_last_move()
            rev = self.playing == 1
            sort_key = lambda x: x[0]
            lst = sorted(lst, key=sort_key, reverse=rev)
            self.successors = [c for _, c in lst]
        self.move_history.append((pos, capturable))
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
        self.remove_sequence_spaces(coord.sub(seq.start, seq.dir), id, Block.HEAD)
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
        self.remove_sequence_spaces(coord.add(seq.end, seq.dir), id, Block.TAIL)
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
        board_slice = SLICE_MAP[dir](self.cells, pos[0], pos[1], Board.sequence_win - 1)
        dist = self.find_stone(board_slice)
        if dist is None:
            return
        new_pos = coord.add(pos, coord.mul(dir, dist))
        for id in self.seq_map.get(new_pos, set()).copy():
            seq = self.seq_list[id]
            if seq.dir == dir or seq.dir == coord.neg(dir):
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
                visited.add(coord.neg(seq.dir))
            elif pos in seq.block_cells:
                self.replace_sequence(id)
                visited.add(seq.dir if pos < seq.start else coord.neg(seq.dir))
            else:
                self.split_sequence_at_removed_stone(id)
                visited.update((seq.dir, coord.neg(seq.dir)))
        for d in visited.symmetric_difference(SLICE_MAP.keys()):
            self.find_and_replace_sequences(pos, d)

    def capturable_stones(self, pos: Coord, cases: tuple[tuple[int]]) -> list[Coord]:
        """
        Returns a list of capturable stones from the given position.
        """
        capturable = []
        for dir in SLICE_MAP.keys():
            board_slice = SLICE_MAP[dir](self.cells, pos[0], pos[1], 4)
            if self.capturable_slice(board_slice, cases):
                capturable.extend(
                    [coord.add(pos, dir), coord.add(pos, coord.add(dir, dir))]
                )
        return capturable

    def get_threats(self, player: int) -> list[Sequence]:
        """
        Returns a list of sequences that are threats that can result in a win
        """
        threats = []
        for seq in self.seq_list:
            if seq.player == player and seq.spaces[0] == 1 and seq.spaces[1] == 0:
                threats.append(seq)
        return threats

    def filter_successors(self, cell: Coord, close: bool = False) -> list[Coord]:
        """
        Filter out successors that are equivalent to the current board
        """
        # neighbors = self.cell_close_neighbors if close else self.cell_neighbors
        # return (c for c in neighbors[cell] if self.can_place(c))
        if close:
            return [c for c in self.cell_close_neighbors[cell] if self.can_place(c)]
        skip = False
        successors = []
        for i, offset in enumerate(NEIGHBORS):
            if skip:
                skip = False
                continue
            c = coord.add(cell, offset)
            if self.can_place(c):
                successors.append(c)
            elif not (i & 1):  # if i is even
                skip = True  # skip the next neighbor because there is an obstacle
        return successors

    def get_threats(self) -> list[tuple[bool, Sequence]]:
        """
        Returns a list of sequences that are threats that can result in a win
        """
        threats = []
        for seq in self.seq_list.values():
            if (n := seq.is_threat(self.capture)) > 0:
                threats.append((n == 2, seq))  # 1 = seq threat, 2 = capture threat
        return threats

    def get_successors(self) -> list[Coord]:
        """
        Returns the coordinates of the neighbor cells that are useful for the heuristic
        """
        successors = set()

        # Case where no ally stones exist on the board
        # Returns the closest neighbor of each enemy stone
        if not any(self.cells[c] == self.playing for c in self.stones):
            for stone in self.stones:
                successors.update(self.filter_successors(stone, close=True))
            successors = set(
                c for c in successors if not self.is_free_double(c, self.playing)
            )
            if successors:
                return list(successors)

        # Case where there are sequences that are threats
        # Returns the forced moves to block/extend the threats
        threats = self.get_threats()
        if threats:
            for is_capture, seq in threats:
                if is_capture:
                    for c in seq.threat_cells:
                        if self.can_place(c):
                            successors.add(c)
                else:
                    if self.capture:
                        count, captures = self.capturable_stones_in_sequences(seq)
                        if count > 0:
                            for c in captures:
                                if self.can_place(c):
                                    successors.add(c)
                    for c in seq.threat_cells:
                        if self.can_place(c):
                            successors.add(c)
            successors = set(
                c for c in successors if not self.is_free_double(c, self.playing)
            )
            if successors:
                return list(successors)

        # Case where there are no threats but there are sequences
        # Returns all cost cells of the sequences
        for seq in self.seq_list.values():
            for c in seq.cost_cells:
                if self.can_place(c):
                    successors.add(c)

        any_sequences = True if self.seq_list else False
        # Case for neighbors that are not related to existing sequences
        # Returns the neighbors of all ally stones in range of 2 if no sequences exist
        # or 1 if sequences exist
        for stone in self.stones:
            if self.cells[stone] == self.playing:
                successors.update(self.filter_successors(stone, close=any_sequences))
        successors = set(
            c for c in successors if not self.is_free_double(c, self.playing)
        )
        if successors:
            return list(successors)
        for stone in self.stones:
            successors.update(self.filter_successors(stone, close=True))
        successors = set(
            c for c in successors if not self.is_free_double(c, self.playing)
        )
        return list(successors)

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
        pos = coord.add(pos, dir)  # skip the first stone
        spaces = 0
        while (
            spaces < Board.sequence_win
            and coord.in_bound(pos, Board.size)
            and self.cells[pos] != opponent
        ):
            pos = coord.add(pos, dir)
            spaces += 1
        return spaces

    def get_half_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        """
        Returns a sequence in a direction from a position.
        """
        board_slice = SLICE_MAP[dir](self.cells, pos[0], pos[1], Board.sequence_win)
        shape = self.slice_to_shape(board_slice)
        seq = Sequence(player, shape, pos, dir)
        seq.spaces = self.get_spaces(seq.end, seq.dir, -seq.player)
        seq.is_blocked = Block.NO if seq.spaces != 0 else Block.tuple_to_block(seq.dir)
        if Block.tuple_to_block(seq.dir) == Block.HEAD:
            seq.start = seq.end
            seq.shape = seq.shape[::-1]
            seq.dir = coord.neg(seq.dir)
        return seq

    def get_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        """
        Returns a sequence that combines the sequences starting from the same cell
        but having opposite directions.
        """
        head = self.get_half_sequence(pos, dir, player)
        tail = self.get_half_sequence(pos, coord.neg(dir), player)
        return head + tail if dir < coord.neg(dir) else tail + head

    def capturable_stones_in_sequences(self, seq: Sequence) -> tuple[int, list[Coord]]:
        """
        Returns whether a sequence can be broken/reduced with a capture and the cell
        that leads to the capture.
        """
        count = 0
        captures = []
        for stone in seq:
            visited = set()
            for id in self.seq_map.get(stone, set()).copy():
                s = self.seq_list[id]
                if seq.id == id or seq.player != s.player:
                    continue
                n, cells = s.capturable_sequence()
                if abs(n) > 0:
                    count += abs(n)
                    captures.extend(cells)
                visited.update((s.dir, coord.neg(s.dir)))
            for d in visited.symmetric_difference(SLICE_MAP.keys()):
                s = self.get_sequence(stone, d, seq.player)
                n, cells = s.capturable_sequence()
                if abs(n) > 0:
                    count += abs(n)
                    captures.extend(cells)
        return count, captures
