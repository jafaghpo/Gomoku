from dataclasses import dataclass
from enum import IntEnum
from collections import namedtuple
from functools import cache
from typing import ClassVar, Iterator
import copy


SEQUENCE_WIN = 5
MAX_SCORE = 1e15
BASE_SCORE = 10
CAPTURE_BASE_SCORE = 5
BLOCK_PENALTY = 4
CAPTURE_WIN = 5
DELTA_WIN = SEQUENCE_WIN - CAPTURE_WIN


class Block(IntEnum):
    """
    Enum for indicating which side of a sequence is blocked by an opponent if any.
    """

    NO = 0
    HEAD = 1
    TAIL = 2
    BOTH = 3


class Coord(namedtuple("Coord", "y x")):
    __slots__ = ()

    def __new__(cls, y, x):
        return super().__new__(cls, y, x)

    def __add__(self, other) -> "Coord":
        return Coord(self.y + other[0], self.x + other[1])

    def __sub__(self, other) -> "Coord":
        return Coord(self.y - other[0], self.x - other[1])

    def __mul__(self, a):
        return Coord(self.y * a, self.x * a)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __eq__(self, other) -> bool:
        return self.y == other[0] and self.x == other[1]

    def __hash__(self):
        return hash((self.y, self.x))

    def __str__(self):
        return f"({self.y}, {self.x})"

    def __repr__(self):
        return f"Coord({self.y}, {self.x})"

    def __neg__(self):
        return Coord(-self.y, -self.x)

    def distance(self, other):
        """
        Returns the distance between two coordinates.
        """
        res = self - other
        return max(abs(res.y), abs(res.x))

    def range(self, dir, len, step=1):
        """
        Works the same as range() but with the Coord class which is a tuple.
        """
        for i in range(0, len, step):
            yield self + dir * i

    def range_with_shape(self, dir, shape, step2d=1, step1d=1):
        """
        Takes a shape (length of subsequences separated by an empty cell) and gets
        the range of Coords for each len in shape while skipping cells in between.
        """
        current = self
        for len in shape:
            for coord in current.range(dir, len, step1d):
                current = coord
                yield coord
            current += dir * (step2d + 1)

    def in_range(self, r) -> bool:
        """
        Returns True if the Coord is between a certain range.
        Can accept multiple formats for the range.
        """
        match r:
            case [(y0, x0), (y1, x1)] | (y0, x0, y1, x1) | ((y0, x0), (y1, x1)):
                return y0 <= self.y < y1 and x0 <= self.x < x1
            case [y, x] | (y, x):
                return 0 <= self.y < y and 0 <= self.x < x
            case (i,) | i if i is int:
                return 0 <= self.y < i and 0 <= self.x < i
            case _:
                return False

    def to_block(self) -> Block:
        """
        Depending on the direction (from top to bottom or bottom to top),
        returns the type of block
        """
        if self.y == -1 or (self.y == 0 and self.x == -1):
            return Block.HEAD
        else:
            return Block.TAIL


DIR_STR = {
    Coord(0, -1): "← (left)",
    Coord(-1, -1): "↖ (up-left)",
    Coord(-1, 0): "↑ (up)",
    Coord(-1, 1): "↗ (up-right)",
    Coord(0, 1): "→ (right)",
    Coord(1, 1): "↘ (down-right)",
    Coord(1, 0): "↓ (down)",
    Coord(1, -1): "↙ (down-left)",
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

    bounds: ClassVar[Coord] = Coord(19, 19)

    @property
    def nb_holes(self) -> int:
        """
        Returns the number of holes in the sequence.
        """

        @cache
        def _nb_holes(shape):
            return len(shape) - 1

        return _nb_holes(self.shape)

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
        return self.start + (self.length - 1) * self.dir

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
            acc += self.dir * subseq_len
            holes.append(acc)
            acc += self.dir
        return tuple(holes)

    @property
    def block_head(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that are the head of a block.
        """
        return (self.start - self.dir,) if self.is_blocked & Block.HEAD else ()

    @property
    def block_tail(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that are the tail of a block.
        """
        return (self.end + self.dir,) if self.is_blocked & Block.TAIL else ()

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
        max_len_head = min(self.spaces[0], max(SEQUENCE_WIN - self.length, 2))
        max_len_tail = min(self.spaces[1], max(SEQUENCE_WIN - self.length, 2))
        head = tuple(self.start + (i + 1) * -self.dir for i in range(max_len_head))
        tail = tuple(self.end + (i + 1) * self.dir for i in range(max_len_tail))
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
        return tuple(self.start.range_with_shape(self.dir, self.shape))

    @property
    def flank_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that flank the sequence.
        """
        return self.filter_in_bounds((self.start - self.dir, self.end + self.dir))

    @property
    def cost_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that directly impact
        the growth of the sequence, meaning the holes and the flanking cells.
        """
        end = min(max(SEQUENCE_WIN - self.length - self.nb_holes, 0), 2)
        head, tail = self.space_cells
        return self.filter_in_bounds(self.holes + head[:end] + tail[:end])

    @property
    def is_dead(self) -> bool:
        """
        Returns True if the sequence cannot win.
        """
        return self.capacity < SEQUENCE_WIN or len(self) < 2

    @property
    def is_win(self) -> bool:
        """
        Returns True if the sequence is winning.
        """
        return max(self.shape) >= SEQUENCE_WIN

    def __str__(self):
        s = f"Sequence {self.id} (p{self.player if self.player == 1 else 2}):\n"
        s += f"  shape: {self.shape}\n"
        s += f"  spaces: {self.spaces}\n"
        s += f"  direction: {DIR_STR[self.dir]}\n"
        s += f"  score: {self.score()}\n"
        s += f"  is blocked: {self.is_blocked.name}\n"
        if self.is_blocked != Block.NO:
            s += f"  block cells: {', '.join(map(str, self.block_cells))}\n"
        s += f"  rest cells: {', '.join(map(str, self.rest_cells))}\n"
        s += f"  cost cells: {', '.join(map(str, self.cost_cells))}\n"
        s += f"  growth cells: {', '.join(map(str, self.growth_cells))}\n"
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

    def __len__(self) -> int:
        @cache
        def _stones(shape):
            return sum(shape)

        return _stones(self.shape)

    def __contains__(self, coord: Coord) -> bool:
        return coord in self.rest_cells

    def __iter__(self) -> Iterator[Coord]:
        return iter(self.rest_cells)

    def copy(self) -> "Sequence":
        return copy.copy(self)

    def score(self, capture: dict[int, int] | None = None) -> int:
        """
        Returns the score of the sequence.
        """

        @cache
        def _capture_score(capture: tuple[tuple[int, int]]) -> int:
            score = 0
            for player, count in capture:
                exponent = max(DELTA_WIN + count, 0)
                if exponent >= SEQUENCE_WIN:
                    return MAX_SCORE * player
                n = CAPTURE_BASE_SCORE ** exponent * player - player
                score += BASE_SCORE * n
            return score

        @cache
        def _score(shape: tuple[int], base: int) -> int:
            if not shape:
                return 0
            n = BASE_SCORE
            for subseq in shape:
                if subseq >= SEQUENCE_WIN:
                    return MAX_SCORE
                n *= base ** subseq
            return n

        capture_score = 0
        shape = self.shape
        if capture:
            tmp_capture = copy.copy(capture)
            x = self.capturable_sequence()
            tmp_capture[-self.player] += abs(x)
            capture_score = _capture_score(tuple(capture.items()))
            
            # capture_shape = max(DELTA_WIN + capture[-self.player], 1)

            # match self.capturable_sequence():
            #     case 1:
            #         capture_score = _score((capture_shape,), CAPTURE_BASE_SCORE)
            #         print(f"capture_score: {capture_score}")
            #         capture_score += self.player
            #         shape = shape[1:]
            #     case -1:
            #         capture_score = _score((capture_shape,), CAPTURE_BASE_SCORE)
            #         capture_score += self.player
            #         shape = shape[:-1]
            #     case 2:
            #         capture_shape = max(DELTA_WIN + capture[-self.player] + 1, 1)
            #         capture_score = _score((capture_shape,), CAPTURE_BASE_SCORE)
                    
            #         shape = shape[1:-1]
            #     case _:
            #         capture_score = 0
        block_penalty = BLOCK_PENALTY if self.is_blocked != Block.NO else 0
        base = BASE_SCORE - self.nb_holes - block_penalty
        seq_score = _score(shape, base) * self.player
        print(f"seq_score: {seq_score}, capture_score: {capture_score}")
        return seq_score + capture_score

    def is_block(self, pos: Coord) -> Block:
        """
        Returns a Block type depending on whether the given position can block
        the sequence if an opponent stone is placed there.
        """
        if pos == self.start - self.dir and self.spaces[0] > 0:
            return Block.HEAD
        elif pos == self.end + self.dir and self.spaces[1] > 0:
            return Block.TAIL
        else:
            return Block.NO

    def filter_in_bounds(self, cells: tuple[Coord]) -> tuple[Coord]:
        """
        Returns cells that are not out of range.
        """
        return tuple(filter(lambda c: c.in_range(self.bounds), cells))

    def extend_tail(self, pos: Coord, space: int) -> None:
        """
        Extends the sequence by one cell.
        """
        if self.end + self.dir == pos:
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
        if self.start - self.dir == pos:
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
            or pos.distance(self.start) <= 2
            or pos.distance(self.end) <= 2
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
                Block(self.is_blocked & Block.HEAD + Block.TAIL),
            ),
            Sequence(
                self.player,
                self.shape[index + 1 :],
                pos + self.dir,
                self.dir,
                (0, self.spaces[1]),
                Block(self.is_blocked & Block.TAIL + Block.HEAD),
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
        self.spaces = (self.spaces[0] + self.start.distance(start), self.spaces[1])
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
        self.spaces = (self.spaces[0], self.spaces[1] + self.end.distance(end))
        if self.is_blocked == Block.TAIL or self.is_blocked == Block.BOTH:
            self.is_blocked = Block(self.is_blocked - Block.TAIL)

    def capturable_sequence(self) -> int:
        """
        Returns whether the sequence is capturable.
        """

        def is_blocked_by_opponent(block: tuple[Coord]) -> bool:
            return block != () and block[0].in_range(self.bounds)

        head = int(is_blocked_by_opponent(self.block_head) and self.shape[0] == 2)
        tail = int(
            is_blocked_by_opponent(self.block_tail)
            and self.nb_holes != 0
            and self.shape[-1] == 2
        )
        if head == 1 and tail == 1:
            return 2
        return head - tail
