from dataclasses import dataclass
from enum import IntEnum
from collections import namedtuple
from functools import cache
from typing import ClassVar


MAX_SEQ_LEN = 5


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

    def range2d(self, dir, shape, step2d=1, step1d=1):
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

    def get_block_dir(self) -> Block:
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


class SeqType(IntEnum):
    # x o o o x
    DEAD = 0
    # x o - o - -
    BLOCKED_HOLED_TWO = 300
    # x o o - - -
    BOARD_BLOCKED_TWO = 500
    BLOCKED_TWO = -1000  # negative value because it can be captured by opponent
    # - o - o - -
    HOLED_TWO = 1450
    # - o o - - -
    TWO = 1500
    # x o - o - o
    BLOCKED_DOUBLE_HOLED_THREE = 1550
    # x o o - o -
    BLOCKED_HOLED_THREE = 1650
    # x o o o - -
    BLOCKED_THREE = 1700
    # o - o - o
    DOUBLE_HOLED_THREE = 1800
    # - o o - o -
    HOLED_THREE = 4900
    # - o o o - -
    THREE = 5000
    # x o o - o o || x o o o - o
    BLOCKED_HOLED_FOUR = 9000
    # x o o o o -
    BLOCKED_FOUR = 10000
    # - o o - o o || - o o o - o
    HOLED_FOUR = 10500
    # - o o o o -
    FOUR = 100000
    # o o o o o
    FIVE = 2e9


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
    def stones(self) -> int:
        """
        Returns the number of stones in the sequence.
        """

        @cache
        def _stones(shape):
            return sum(shape)

        return _stones(self.shape)

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
        return self.stones + self.nb_holes

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
    def block_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that block the sequence.
        """
        cells = ()
        if self.is_blocked == Block.NO:
            return ()
        elif self.is_blocked == Block.HEAD:
            cells = (self.start - self.dir,)
        elif self.is_blocked == Block.TAIL:
            cells = (self.end + self.dir,)
        else:
            cells = (self.start - self.dir, self.end + self.dir)
        return cells

    @property
    def space_cells(self) -> tuple[tuple[Coord]]:
        """
        Returns the coordinates of the cells that are empty around the sequence.
        """
        max_len_head = min(self.spaces[0], max(MAX_SEQ_LEN - self.length, 2))
        max_len_tail = min(self.spaces[1], max(MAX_SEQ_LEN - self.length, 2))
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
        return tuple(self.start.range2d(self.dir, self.shape))

    @property
    def cost_cells(self) -> tuple[Coord]:
        """
        Returns the coordinates of the cells that directly impact
        the growth of the sequence, meaning the holes and the flanking cells.
        """
        cells = self.holes
        if self.length >= MAX_SEQ_LEN:
            return self.filter_in_bounds(cells)
        elif self.is_blocked == Block.NO:
            cells += (self.start - self.dir, self.end + self.dir)
        elif self.is_blocked == Block.HEAD:
            cells += (self.end + self.dir,)
        elif self.is_blocked == Block.TAIL:
            cells += (self.start - self.dir,)
        return self.filter_in_bounds(cells)

    @property
    def is_dead(self) -> bool:
        """
        Returns True if the sequence cannot win.
        """
        return self.capacity < MAX_SEQ_LEN or self.stones < 2

    @property
    def type(self) -> SeqType:
        """
        Returns the type of the sequence and its value based on the number of stones,
        the number of holes, and the maximum achievable length of the sequence.
        """

        @cache
        def _type(
            stones: int, holes: int, is_blocked: Block, is_dead: bool, op_block: bool
        ) -> SeqType:
            if is_dead:
                return SeqType.DEAD
            match (stones, holes, is_blocked):
                case (2, 1, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_HOLED_TWO
                case (2, 0, Block.HEAD | Block.TAIL) if not op_block:
                    return SeqType.BOARD_BLOCKED_TWO
                case (2, 0, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_TWO
                case (2, 1, Block.NO):
                    return SeqType.HOLED_TWO
                case (2, 0, Block.NO):
                    return SeqType.TWO
                case (3, 2, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_DOUBLE_HOLED_THREE
                case (3, 1, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_HOLED_THREE
                case (3, 0, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_THREE
                case (3, 2, Block.NO):
                    return SeqType.DOUBLE_HOLED_THREE
                case (3, 1, Block.NO):
                    return SeqType.HOLED_THREE
                case (3, 0, Block.NO):
                    return SeqType.THREE
                case (4, 1, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_HOLED_FOUR
                case (4, 0, Block.HEAD | Block.TAIL):
                    return SeqType.BLOCKED_FOUR
                case (4, 1, Block.NO):
                    return SeqType.HOLED_FOUR
                case (4, 0, Block.NO):
                    return SeqType.FOUR
                case (s, 0, _) if s >= 5:
                    return SeqType.FIVE
                case _:
                    return SeqType.DEAD

        is_dead = self.capacity < MAX_SEQ_LEN
        # op_block stands for opponent's block
        op_block = any(map(lambda c: c.in_range(self.bounds), self.block_cells))
        return _type(self.stones, self.nb_holes, self.is_blocked, is_dead, op_block)

    def __str__(self):
        s = f"Sequence {self.id} (p{self.player}):\n"
        s += f"  stone count: {self.stones}\n"
        s += f"  length (including spaces): {self.length}\n"
        s += f"  capacity: {self.capacity}\n"
        s += f"  shape: {self.shape}\n"
        s += f"  start: {self.start}\n"
        s += f"  end: {self.end}\n"
        s += f"  direction: {DIR_STR[self.dir]}\n"
        s += f"  spaces: {self.spaces}\n"
        s += f"  is_blocked: {self.is_blocked.name}\n"
        s += f"  block_cells: {', '.join(map(str, self.block_cells))}\n"
        s += f"  type: {self.type.name} (score: {self.type})\n"
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
        if self.start > other.start:
            self, other = (other, self)
        self.shape = (
            self.shape[:-1] + (self.shape[-1] + other.shape[0] - 1,) + other.shape[1:]
        )
        self.is_blocked = Block(self.is_blocked + other.is_blocked)
        self.spaces = (self.spaces, other.spaces)
        return self

    def can_pos_block(self, pos: Coord) -> Block:
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

    def split(self, pos: Coord) -> tuple["Sequence", "Sequence"]:
        """
        Splits the sequence at the given position.
        """
        index = self.holes.index(pos)
        return (
            Sequence(
                self.player,
                self.shape[: index + 1],
                self.start,
                self.dir,
                Block(self.is_blocked + Block.TAIL),
                (self.spaces[0], 0),
            ),
            Sequence(
                self.player,
                self.shape[index + 1 :],
                pos + self.dir,
                self.dir,
                Block(self.is_blocked + Block.HEAD),
                self.spaces,
            ),
        )
