from dataclasses import dataclass
from enum import IntEnum
from collections import namedtuple
from functools import cache


MAX_SEQ_LEN = 5

class Block(IntEnum):
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
        return self.y == other.y and self.x == other.x

    def __hash__(self):
        return hash((self.y, self.x))

    def __str__(self):
        return f"({self.y}, {self.x})"

    def __repr__(self):
        return f"Coord({self.y}, {self.x})"

    def __neg__(self):
        return Coord(-self.y, -self.x)

    def range(self, dir, len, step=1):
        for i in range(0, len, step):
            yield self + dir * i

    def range2d(self, dir, shape, step2d=1, step1d=1):
        current = self
        for len in shape:
            for coord in current.range(dir, len, step1d):
                current = coord
                yield coord
            current += dir * (step2d + 1)

    def in_range(self, r) -> bool:
        match r:
            case [(y0, x0), (y1, x1)] | (y0, x0, y1, x1) | ((y0, x0), (y1, x1)):
                return y0 <= self.y <= y1 and x0 <= self.x <= x1
            case [y, x] | (y, x):
                return 0 <= self.y < y and 0 <= self.x < x
            case (i,) | i if i is int:
                return 0 <= self.y < i and 0 <= self.x < i
            case _:
                return False
    
    def get_block_dir(self) -> Block:
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
    BLOCKED_TWO = -1000 # negative value because it can be captured by opponent
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
    A sequence is a set of cells that are aligned in some way.
    """

    player: int
    shape: tuple[int]
    start: Coord
    dir: Coord
    spaces: tuple[int] = (0, 0)
    is_blocked: Block = Block.NO
    id: int = -1

    @property
    def stones(self) -> int:
        @cache
        def stones(shape):
            return sum(shape)
        return stones(self.shape)
    
    @property
    def nb_holes(self) -> int:
        @cache
        def nb_holes(shape):
            return len(shape) - 1
        return nb_holes(self.shape)

    @property
    def length(self) -> int:
        return self.stones + self.nb_holes

    @property
    def capacity(self) -> int:
        return self.length + sum(self.spaces)

    @property
    def end(self) -> Coord:
        return self.start + (self.length - 1) * self.dir

    @property
    def blocks(self) -> tuple[Coord]:
        if self.is_blocked == Block.NO:
            return ()
        elif self.is_blocked == Block.HEAD:
            return (self.start - self.dir,)
        elif self.is_blocked == Block.TAIL:
            return (self.end + self.dir,)
        else:
            return (self.start - self.dir, self.end + self.dir)

    @property
    def holes(self) -> tuple[Coord]:
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
    def growth_cells(self) -> tuple[tuple[Coord]]:
        head = tuple(
            self.start + (i + 1) * -self.dir
                for i in range(min(self.spaces[0], MAX_SEQ_LEN - self.length))
        )
        tail = tuple(self.end + (i + 1) * self.dir
            for i in range(min(self.spaces[1], MAX_SEQ_LEN - self.length)))
        return head, tail

    @property
    def rest_cells(self) -> tuple[Coord]:
        return self.start.range2d(self.dir, self.shape)

    @property
    def cost_cells(self) -> tuple[Coord]:
        cells = self.holes
        if self.is_blocked == Block.NO:
            return cells + (self.start - self.dir, self.end + self.dir)
        elif self.is_blocked == Block.HEAD:
            return cells + (self.end + self.dir,)
        elif self.is_blocked == Block.TAIL:
            return cells + (self.start - self.dir,)
        else:
            return ()
    
    @property
    def type(self) -> SeqType:
        if self.capacity < MAX_SEQ_LEN:
            return SeqType.DEAD
        match (self.stones, self.nb_holes, self.is_blocked):
            case (2, 1, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_HOLED_TWO
            case (2, 0, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_TWO
            case (2, 1, Block.NO): return SeqType.HOLED_TWO
            case (2, 0, Block.NO): return SeqType.TWO
            case (3, 2, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_DOUBLE_HOLED_THREE
            case (3, 1, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_HOLED_THREE
            case (3, 0, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_THREE
            case (3, 2, Block.NO): return SeqType.DOUBLE_HOLED_THREE
            case (3, 1, Block.NO): return SeqType.HOLED_THREE
            case (3, 0, Block.NO): return SeqType.THREE
            case (4, 1, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_HOLED_FOUR
            case (4, 0, Block.HEAD | Block.TAIL): return SeqType.BLOCKED_FOUR
            case (4, 1, Block.NO): return SeqType.HOLED_FOUR
            case (4, 0, Block.NO): return SeqType.FOUR
            case (s, 0, _) if s >= 5: return SeqType.FIVE
            case _ : return SeqType.DEAD

    def __str__(self):
        s = f"Sequence {self.id} (p{self.player}):\n"
        s += f"  stone count: {self.stones}\n"
        s += f"  length (including spaces): {self.length}\n"
        s += f"  shape: {self.shape}\n"
        s += f"  start: {self.start}\n"
        s += f"  dir: {DIR_STR[self.dir]}\n"
        s += f"  spaces: {self.spaces}\n"
        s += f"  is_blocked: {self.is_blocked.name}\n"
        s += f"  blocks: {', '.join(map(str, self.blocks))}\n"
        s += f"  type: {self.type.name} (score: {self.type})\n"
        s += f"  rest cells: {', '.join(map(str, self.rest_cells))}\n"
        s += f"  cost cells: {', '.join(map(str, self.cost_cells))}\n"
        s += f"  growth cells: {', '.join(map(str, self.growth_cells[0] + self.growth_cells[1]))}\n"
        return s

    def __repr__(self):
        s = f"Sequence(player={self.player}, shape={self.shape}, start={self.start}, "
        s += f"dir={DIR_STR[self.dir]}, is_blocked={self.is_blocked}, id={self.id}, "
        s += f"spaces={self.spaces})"
        return s

    def __add__(self, other):
        self, other = (
            (self, other)
            if self.dir.get_block_dir() == Block.HEAD
            else (other, self)
        )
        self.shape = (self.shape[0] + other.shape[0] - 1,) + self.shape[::-1] + other.shape[1:]
        self.dir = other.dir
        self.is_blocked = Block(self.is_blocked + other.is_blocked)
        self.spaces = tuple(map(sum, zip(self.spaces, other.spaces)))
        return self