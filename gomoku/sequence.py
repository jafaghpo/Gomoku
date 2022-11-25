from dataclasses import dataclass
from enum import IntEnum
# from collections import namedtuple
from functools import cache
from typing import ClassVar, Iterator
import copy
import gomoku.coord as coord

MAX_SCORE = int(1e9)
BASE_SCORE = 10
CAPTURE_BASE_SCORE = BASE_SCORE + 1
BLOCK_PENALTY = BASE_SCORE // 4

Coord = tuple[int, int]


class Block(IntEnum):
    """
    Enum for indicating which side of a sequence is blocked by an opponent if any.
    """

    NO = 0
    HEAD = 1
    TAIL = 2
    BOTH = 3

    @staticmethod
    def tuple_to_block(t: Coord) -> "Block":
        """
        Returns the Block enum value corresponding to the tuple.
        """
        return Block.HEAD if t[0] == -1 or (t[0] == 0 and t[1] == -1) else Block.TAIL



DIR_STR = {
    (0, -1): "←",
    (-1, -1): "↖",
    (-1, 0): "↑",
    (-1, 1): "↗",
    (0, 1): "→",
    (1, 1): "↘",
    (1, 0): "↓",
    (1, -1): "↙",
}


@dataclass(slots=True)
class Sequence:
    """
    A sequence is a set of cells that are aligned in a line.
    - player: the id of the player who owns the sequence
    - shape: a list of lengths of subsequences separated by an empty cell.
    - start: the coordinates of the starting cell of the sequence
    - dir: the direction of the sequence
    - spaces: the number of empty cells or ally cells around the sequence
    - is_blocked: indicates if the sequence is blocked from the sides
    - id: the id of the sequence
    """

    player: int
    shape: tuple[int]
    start: Coord
    dir: Coord
    spaces: tuple[int] = (0, 0)
    is_blocked: Block = Block.NO
    id: int = -1

    board_size: ClassVar[int] = 19
    sequence_win: ClassVar[int] = 5
    capture_win: ClassVar[int] = 5
    capture_weight: ClassVar[int] = 1

    @property
    def nb_holes(self) -> int:
        """
        Returns the number of holes in the sequence.
        """
        return len(self.shape) - 1

    @property
    def length(self) -> int:
        """
        Returns the length of the sequence including the empty cells inside.
        """
        return len(self) + self.nb_holes

    @property
    def capacity(self) -> int:
        """
        Returns the maximum length the sequence can achieve.
        """
        return self.length + sum(self.spaces)

    @property
    def end(self) -> Coord:
        """
        Returns the coordinates of the last cell of the sequence.
        """
        return coord.add(self.start, coord.mul(self.dir, self.length - 1))

    @property
    def holes(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that are holes in the sequence.
        """
        if self.nb_holes == 0:
            return ()
        holes = []
        acc = self.start
        for subseq_len in self.shape[:-1]:
            acc = coord.add(acc, coord.mul(self.dir, subseq_len))
            holes.append(acc)
            acc = coord.add(acc, self.dir)
        return tuple(holes)

    @property
    def block_head(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that are the head of a block.
        """
        is_blocked_head = self.is_blocked & Block.HEAD
        return (coord.sub(self.start, self.dir),) if is_blocked_head else ()

    @property
    def block_tail(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that are the tail of a block.
        """
        is_blocked_tail = self.is_blocked & Block.TAIL
        return (coord.add(self.end, self.dir),) if is_blocked_tail else ()

    @property
    def block_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that block the sequence.
        """
        return self.block_head + self.block_tail

    @property
    def space_cells(self) -> tuple[tuple[Coord]]:
        """
        Returns the coordinates of the cells that are empty around the sequence.
        """
        max_len_head = min(self.spaces[0], max(Sequence.sequence_win - self.length, 2))
        max_len_tail = min(self.spaces[1], max(Sequence.sequence_win - self.length, 2))
        head = tuple(coord.add(self.start, coord.mul(coord.neg(self.dir), i + 1)) for i in range(max_len_head))
        tail = tuple(coord.add(self.end, coord.mul(self.dir, i + 1)) for i in range(max_len_tail))
        return self.filter_in_bounds(head), self.filter_in_bounds(tail)

    @property
    def growth_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that can grow the sequence.
        """
        return self.space_cells[0] + self.space_cells[1] + self.holes

    @property
    def rest_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the stones in the sequence.
        """
        return tuple(coord.range_shape(self.start, self.dir, self.shape))

    @property
    def flank_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that flank the sequence.
        """
        match self.is_blocked:
            case Block.NO:
                return coord.sub(self.start, self.dir), coord.add(self.end, self.dir)
            case Block.HEAD:
                return coord.add(self.end, self.dir),
            case Block.TAIL:
                return coord.sub(self.start, self.dir),
            case Block.BOTH:
                return (),

    @property
    def cost_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that directly impact
        the growth of the sequence, meaning the holes and the flanking cells.
        """
        end = min(max(Sequence.sequence_win - self.length, 0), 2)
        head, tail = self.space_cells
        return self.filter_in_bounds(self.holes + head[:end] + tail[:end])

    @property
    def is_dead(self) -> bool:
        """
        Returns True if the sequence cannot win.
        """
        return self.capacity < Sequence.sequence_win or len(self) < 2

    @property
    def is_win(self) -> bool:
        """
        Returns True if the sequence is winning.
        """
        return max(self.shape) >= Sequence.sequence_win
    
    def to_string(self, playing: int, capture: dict[int, int] | None = None) -> str:
        """
        Returns a string representation of the sequence.
        """
        s = f"[{self.id}]\t"
        s += f"\u001b[{'32' if self.player == 1 else '36'}m"
        s += f"Player {1 if self.player == 1 else 2: <4}"
        s += f"start: {str(self.start): <10}dir: {str(DIR_STR[self.dir]): <4}"
        s += f"block: {self.is_blocked.name.lower(): <8}"
        s += f"shape: {str(self.shape): <20}score: {self.score(playing, capture)}"
        s += f"\u001b[0m"
        return s

    def __repr__(self):
        s = f"Sequence(player={self.player}, shape={self.shape}, start={self.start}, "
        s += f"dir={DIR_STR[self.dir]}, is_blocked={self.is_blocked}, id={self.id}, "
        s += f"spaces={self.spaces})"
        return s

    def __add__(self, other):
        self.shape = (
            self.shape[:-1] + (self.shape[-1] + other.shape[0] - 1,) + other.shape[1:]
        )
        self.is_blocked = Block(self.is_blocked + other.is_blocked)
        self.spaces = (self.spaces, other.spaces)
        return self

    def __hash__(self) -> int:
        return hash(
            (
                self.player,
                self.shape,
                self.start,
                self.dir,
                self.spaces,
                self.is_blocked,
            )
        )

    def __eq__(self, other) -> bool:
        return (
            self.player == other.player
            and self.shape == other.shape
            and self.start == other.start
            and self.dir == other.dir
            and self.spaces == other.spaces
            and self.is_blocked == other.is_blocked
        )
    
    def __lt__(self, other) -> bool:
        return self.start < other.start

    def __len__(self) -> int:
        return sum(self.shape)

    def __contains__(self, coord: Coord) -> bool:
        return coord in self.rest_cells

    def __iter__(self) -> Iterator[Coord]:
        return iter(self.rest_cells)

    def copy(self) -> "Sequence":
        return copy.copy(self)

    @staticmethod
    @cache
    def capture_score(capture: int) -> int:
        if capture == 0:
            return 0
        exponent = max(Sequence.sequence_win - Sequence.capture_win + capture, 2)
        if exponent >= Sequence.sequence_win:
            return MAX_SCORE
        return CAPTURE_BASE_SCORE**exponent

    @staticmethod
    @cache
    def seq_score(shape: tuple[int], base: int) -> int:
        if not shape:
            return 0
        n = 1
        for subseq in shape:
            if subseq >= Sequence.sequence_win:
                return MAX_SCORE
            n *= base**subseq
        return n

    def score(self, playing: int, capture: dict[int, int] | None = None) -> int:
        """
        Returns the score of the sequence.
        """
        capture_score = 0
        shape = self.shape
        if capture and self.player != playing and self.nb_holes == 0:
            tmp_capture = copy.copy(capture)
            n, _ = self.capturable_sequence()
            if n != 0:
                tmp_capture[playing] += abs(n)
                capture_score = Sequence.capture_score(tmp_capture[playing]) * playing
                capture_score *= Sequence.capture_weight
            if n == 1:
                shape = shape[1:]
            elif n == -1:
                shape = shape[:-1]
            elif n == 2:
                shape = shape[1:-1]
            if not shape:
                return capture_score
        block_penalty = BLOCK_PENALTY * int(self.is_blocked != Block.NO)
        base = max(BASE_SCORE - self.nb_holes - block_penalty, 2)
        seq_score = Sequence.seq_score(shape, base) * self.player
        return seq_score + capture_score

    def is_block(self, pos: Coord) -> Block:
        """
        Returns a Block type depending on whether the given position can block
        the sequence if an opponent stone is placed there.
        """
        if pos == coord.sub(self.start, self.dir) and self.spaces[0] > 0:
            return Block.HEAD
        elif pos == coord.add(self.end, self.dir) and self.spaces[1] > 0:
            return Block.TAIL
        else:
            return Block.NO

    def filter_in_bounds(self, cells: tuple[Coord]) -> tuple[Coord]:
        """
        Returns cells that are not out of range.
        """
        return tuple(c for c in cells if coord.in_bound(c, Sequence.board_size))

    def extend_tail(self, pos: Coord, space: int) -> None:
        """
        Extends the sequence by one cell.
        """
        if coord.add(self.end, self.dir) == pos:
            self.shape = self.shape[:-1] + (self.shape[-1] + 1,)
            self.spaces = (self.spaces[0], space)
        else:
            self.shape += (1,)
            self.spaces = (self.spaces[0], space)
        if self.spaces[1] == 0:
            self.is_blocked = Block(self.is_blocked + Block.TAIL)

    def extend_head(self, pos: Coord, space: int) -> None:
        """
        Extends the sequence by one cell.
        """
        if coord.sub(self.start, self.dir) == pos:
            self.shape = (self.shape[0] + 1,) + self.shape[1:]
            self.spaces = (space, self.spaces[1])
        else:
            self.shape = (1,) + self.shape
            self.spaces = (space, self.spaces[1])
        self.start = pos
        if self.spaces[0] == 0:
            self.is_blocked = Block(self.is_blocked + Block.HEAD)

    def extend_hole(self, pos: Coord) -> None:
        """
        Extends the sequence by one, filling a hole in the sequence.
        """
        if not pos in self.holes:
            return
        index = self.holes.index(pos)
        self.shape = (
            self.shape[:index]
            + (self.shape[index] + self.shape[index + 1] + 1,)
            + self.shape[index + 2 :]
        )

    def can_extend(self, pos: Coord) -> bool:
        """
        Returns whether the sequence can be extended by one cell.
        """
        return (
            self.start < pos < self.end
            or coord.distance(pos, self.start) <= 2
            or coord.distance(pos, self.end) <= 2
        )

    def split_at_blocked_hole(self, pos: Coord) -> tuple["Sequence", "Sequence"]:
        """
        Splits the sequence at the position where an enemy stone filled a hole.
        """
        index = self.holes.index(pos)
        return (
            Sequence(
                self.player,
                self.shape[: index + 1],
                self.start,
                self.dir,
                (self.spaces[0], 0),
                Block((self.is_blocked & Block.HEAD) + Block.TAIL),
            ),
            Sequence(
                self.player,
                self.shape[index + 1 :],
                coord.add(pos, self.dir),
                self.dir,
                (0, self.spaces[1]),
                Block((self.is_blocked & Block.TAIL) + Block.HEAD),
            ),
        )

    def reduce_head(self) -> None:
        """
        Reduces sequence head by one cell.
        """
        start = self.start
        self.start = self.rest_cells[1]
        self.shape = (self.shape[0] - 1,) + self.shape[1:]
        if self.shape[0] == 0:
            self.shape = self.shape[1:]
        self.spaces = (self.spaces[0] + coord.distance(self.start, start), self.spaces[1])
        if self.is_blocked == Block.HEAD or self.is_blocked == Block.BOTH:
            self.is_blocked = Block(self.is_blocked - Block.HEAD)

    def reduce_tail(self) -> None:
        """
        Reduces sequence tail by one cell.
        """
        end = self.end
        self.shape = self.shape[:-1] + (self.shape[-1] - 1,)
        if self.shape[-1] == 0:
            self.shape = self.shape[:-1]
        self.spaces = (self.spaces[0], self.spaces[1] + coord.distance(self.end, end))
        if self.is_blocked == Block.TAIL or self.is_blocked == Block.BOTH:
            self.is_blocked = Block(self.is_blocked - Block.TAIL)

    def capturable_sequence(self) -> tuple[int, tuple[Coord]]:
        """
        Returns whether the sequence is capturable and the cell that can capture it
        """
        head = (self.is_blocked & Block.HEAD) != 0 and self.shape[0] == 2
        tail = (self.is_blocked & Block.TAIL) != 0 and self.shape[-1] == 2
        match (head, tail):
            case (False, False):
                return 0, ()
            case (True, False):
                return 1, self.flank_cells if not self.nb_holes else (self.holes[0],)
            case (False, True):
                return -1, self.flank_cells if not self.nb_holes else (self.holes[-1],)
            case (True, True):
                match self.nb_holes:
                    case 0: return 0, ()
                    case 1: return 2, self.holes
                    case _: return 2, (self.holes[0], self.holes[-1])

    def is_threat(self, capture: dict[int, int] | None) -> int:
        """
        Returns the threat level of the sequence.
        """
        if self.is_dead:
            return 0
        if self.length > Sequence.sequence_win and len(self) >= Sequence.sequence_win - 1:
            return 1
        if len(self) >= Sequence.sequence_win - 1:
            return 1
        if len(self) >= Sequence.sequence_win - 2 and self.is_blocked == Block.NO:
            return 1
        if capture:
            n, _ = self.capturable_sequence()
            if abs(n) != 0 and capture[-self.player] + abs(n) >= Sequence.capture_win:
                return 2
        return 0
                