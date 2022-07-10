from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from collections import namedtuple

MAX_SEQUENCE_STONES = 5


class Block(IntEnum):
    NO = 0
    HEAD = 1
    TAIL = 2
    BOTH = 3


# UNUSED
class SeqType(IntEnum):
    # Five in a row.
    # o o o o o
    FIVE = 5
    # Four in a row with open ends.
    # - o o o o -
    STRAIGHT_FOUR = 4
    # Four pieces in a line of 5 cells.
    # x o o o o -
    # x o o o - o
    # o o - o o x
    FOUR = 3
    # Three pieces in a row.
    # - o o o -
    THREE = 2
    # Three pieces in a line of 5 cells that aren't in a row.
    # - o o - o -
    # - o - o o -
    BROKEN_THREE = 1


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

    def range(self, direction, len, step=1):
        for i in range(0, len, step):
            yield self + direction * i

    def range2d(self, direction, shape, step2d=1, step1d=1):
        current = self
        for len in shape:
            for coord in current.range(direction, len, step1d):
                current = coord
                yield coord
            current += direction * (step2d + 1)

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

    def get_block(self) -> Block:
        if self.y == -1 or (self.y == 0 and self.x == -1):
            return Block.HEAD
        else:
            return Block.TAIL


NEIGHBORS_OFFSET = tuple(
    map(
        Coord._make,
        (
            (-2, -2),
            (-2, 0),
            (-2, 2),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -2),
            (0, -1),
            (0, 1),
            (0, 2),
            (1, -1),
            (1, 0),
            (1, 1),
            (2, -2),
            (2, 0),
            (2, 2),
        ),
    )
)


DIRECTIONS = tuple(map(Coord._make, ((0, -1), (-1, -1), (-1, 0), (-1, 1))))
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
    A sequence is a set of cells that are aligned in some way.
    """

    player: int
    shape: list[int]
    start: Coord
    direction: Coord
    spaces: tuple[int]
    block: Block = Block.NO
    id: int = -1

    @property
    def stones(self) -> int:
        return sum(self.shape)

    @property
    def length(self) -> int:
        return self.stones + len(self.shape) - 1

    @property
    def capacity(self) -> int:
        return self.length + sum(self.spaces)

    @property
    def end(self) -> Coord:
        return self.start + (self.length - 1) * self.direction

    @property
    def holes(self) -> tuple[Coord]:
        if len(self.shape) == 1:
            return ()
        holes = []
        acc = self.start
        for subseq_len in self.shape[:-1]:
            acc += self.direction * subseq_len
            holes.append(acc)
            acc += self.direction
        return tuple(holes)

    @property
    def growth_cells(self) -> tuple[Coord]:
        head = tuple(
            self.start + (i + 1) * -self.direction for i in range(self.spaces[0])
        )
        tail = tuple(self.end + (i + 1) * self.direction for i in range(self.spaces[1]))
        return head + tail

    @property
    def rest_cells(self) -> tuple[Coord]:
        return self.start.range2d(self.direction, self.shape)

    @property
    def cost_cells(self) -> tuple[Coord]:
        cells = self.holes
        if self.block == Block.NO:
            if self.stones >= MAX_SEQUENCE_STONES - 1:
                return cells
            else:
                return cells + (self.start - self.direction, self.end + self.direction)
        elif self.block == Block.HEAD:
            return cells + (self.end + self.direction,)
        elif self.block == Block.TAIL:
            return cells + (self.start - self.direction,)
        else:
            return ()

    def __str__(self):
        s = f"Sequence {self.id} (p{self.player}):\n"
        s += f"  stone count: {self.stones}\n"
        s += f"  length (including spaces): {self.length}\n"
        s += f"  shape: {self.shape}\n"
        s += f"  start: {self.start}\n"
        s += f"  direction: {DIR_STR[self.direction]}\n"
        s += f"  spaces: {self.spaces}\n"
        s += f"  block: {self.block.name}\n"
        s += f"  rest cells: {', '.join(map(str, self.rest_cells))}\n"
        s += f"  cost cells: {', '.join(map(str, self.cost_cells))}\n"
        s += f"  growth cells: {', '.join(map(str, self.growth_cells))}"
        return s

    def __repr__(self):
        s = f"Sequence(player={self.player}, shape={self.shape}, start={self.start}, "
        s += f"direction={DIR_STR[self.direction]}, block={self.block}, id={self.id}, "
        s += f"spaces={self.spaces})"
        return s

    def __add__(self, other):
        self, other = (
            (self, other) if self.direction.get_block() == Block.HEAD else (other, self)
        )
        self.shape[0] += other.shape[0] - 1
        self.shape = self.shape[::-1] + other.shape[1:]
        self.direction = other.direction
        self.block = Block(self.block + other.block)
        self.spaces = tuple(map(sum, zip(self.spaces, other.spaces)))
        return self

    # UNUSED
    def get_block_pos(self, block: Block) -> tuple[Coord]:
        if block == Block.NO:
            return ()
        elif block == Block.HEAD:
            return (self.start - self.direction,)
        elif block == Block.TAIL:
            return (self.end + self.direction,)
        else:
            return (self.start - self.direction, self.end + self.direction)


# UNUSED
class Cells(IntEnum):
    GROWTH = 0  # Empty Cells that can grow a sequence
    REST = 1  # Cells that are part of a sequence
    COST = 2  # Empty Cells that counter the growth of a sequence
    BLOCK = 3  # Cells filled by enemy pieces that block a sequence


@dataclass(init=False, repr=True, slots=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Coord, list[(Cells, int)]]
    seq_list: dict[int, Sequence]
    last_seq_id: int
    last_move: Coord | None
    stones: set[Coord]

    def __init__(self, shape: tuple[int, int] = (19, 19)) -> None:
        self.cells = np.zeros(shape, dtype=np.uint8)
        self.seq_map = {}
        self.seq_list = {}
        self.stones = set()
        self.last_move = None
        self.last_seq_id = 0

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "X", 2: "O"}
        return (
            "\n".join(map(lambda cell: player_repr[cell], row) for row in self.cells),
        )

    def get_pos_c4(self, x: int):
        for y in range(5, -1, -1):
            if self.cells[y, x] == 0:
                print(x, y)
                return Coord(y, x)

    def can_place_c4(self, x: int) -> bool:
        print(x)
        print(self.cells[(x, 0)])
        return self.cells[(x, 0)] == 0

    def can_place(self, pos: Coord) -> bool:
        return pos.in_range(self.cells.shape) and self.cells[pos] == 0

    def add_move(self, pos: Coord, player: int) -> None:
        self.cells[pos] = player
        self.stones.add(pos)
        self.last_move = pos
        print(f"Adding move {pos}")
        seq_list = self.search_sequences(pos, player)
        print("Sequences found:")
        for seq in seq_list:
            print(seq)

    def get_neighbors(self) -> set[Coord]:
        children = set()
        for stone in self.stones:
            raw_neighbors = map(lambda offset: stone + offset, NEIGHBORS_OFFSET)
            neighbors = list(filter(lambda n: self.can_place(n), raw_neighbors))
            children.update(neighbors)
        return children

    def half_sequence(self, current: Coord, dir: Coord, player: int) -> Sequence:
        shape = []
        start = current
        current += dir
        block = Block.NO
        sub_len = 1
        spaces = [0, 0]
        block_dir = dir.get_block()
        idx = 0 if block_dir == Block.HEAD else 1
        empty = False
        while current.in_range(self.cells.shape):
            match (self.cells[current], empty):
                case (0, True):
                    while (
                        current.in_range(self.cells.shape)
                        and self.cells[current] != player ^ 3
                    ):
                        current += dir
                        spaces[idx] += 1
                    return Sequence(player, shape, start, dir, tuple(spaces), block)
                case (0, False):
                    empty = True
                    spaces[idx] += 1
                    if sub_len != 0:
                        shape.append(sub_len)
                        sub_len = 0
                case (p, _) if p == player:
                    spaces[idx] = 0
                    sub_len += 1
                    start = current
                    empty = False
                case (p, _) if p != player:
                    spaces[idx] = 0
                    if sub_len != 0:
                        shape.append(sub_len)
                    block = block_dir
                    return Sequence(player, shape, start, dir, tuple(spaces), block)
            current += dir
        block = block_dir
        if sub_len != 0:
            shape.append(sub_len)
        return Sequence(player, shape, start, dir, tuple(spaces), block)

    def search_sequences(self, pos: Coord, player: int) -> list[Sequence]:
        sequences = []
        for dir in DIRECTIONS:
            head_seq = self.half_sequence(pos, dir, player)
            tail_seq = self.half_sequence(pos, -dir, player)
            sequence = head_seq + tail_seq
            if sequence.length >= 2:
                sequence.id = self.last_seq_id
                self.last_seq_id += 1
                sequences.append(sequence)
        return sequences


# TODO:
# - Store generated sequences in seq_map & seq_list
# - Divide too large sequences:
#   extract the biggest sub sequence inferior to the max sequence length and repeat with
#   the rest of the sequence until you have a sequence inferior to max seq len
# - Use cache for the Sequence properties that take non-negligeable time when repeated.
