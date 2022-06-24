from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from collections import namedtuple


class Block(IntEnum):
    NO = 0
    HEAD = 1
    TAIL = 2
    BOTH = 3

    def __str__(self):
        match self:
            case Block.NO:
                return "None"
            case Block.HEAD:
                return "Head"
            case Block.TAIL:
                return "Tail"
            case Block.BOTH:
                return "Both"


class SeqType(IntEnum):
    # Five in a row.
    # o o o o o
    FIVE = 5
    # Four in a row with open ends.
    # - o o o o -
    STRAIGHT_FOUR = 4
    # Four pieces in a line of 5 squares.
    # x o o o o -
    # x o o o - o
    # o o - o o x
    FOUR = 3
    # Three pieces in a row.
    # - o o o -
    THREE = 2
    # Three pieces in a line of 5 squares that aren't in a row.
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
    block: Block = Block.NO
    capacity: int = 0
    id: int = -1

    @property
    def length(self) -> int:
        return sum(self.shape) + len(self.shape) - 1

    @property
    def end(self) -> Coord:
        return self.start + (self.length - 1) * self.direction

    def __str__(self):
        s = f"Sequence {self.id} (player{self.player}):\n"
        s += f"  shape: {self.shape}\n"
        s += f"  start: {self.start}\n"
        s += f"  direction: {DIR_STR[self.direction]}\n"
        s += f"  block: {self.block}\n"
        s += f"  capacity: {self.capacity}\n"
        return s

    def __repr__(self):
        s = f"Sequence(player={self.player}, shape={self.shape}, start={self.start}, "
        s += f"direction={DIR_STR[self.direction]}, block={self.block}, id={self.id}, "
        s += f"capacity={self.capacity})"
        return s

    def __add__(self, other):
        self, other = (
            (self, other) if self.direction.get_block() == Block.HEAD else (other, self)
        )
        self.shape[0] += other.shape[0] - 1
        self.shape = self.shape[::-1] + other.shape[1:]
        self.direction = other.direction
        self.block += other.block
        self.capacity += other.capacity - 1
        return self


@dataclass(init=False, repr=True, slots=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Coord, list[int]]
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
        capacity = sub_len = 1
        empty = False
        while current.in_range(self.cells.shape):
            match (self.cells[current], empty):
                case (0, True):
                    while (
                        current.in_range(self.cells.shape)
                        and self.cells[current] != player ^ 3
                    ):
                        current += dir
                        capacity += 1
                    return Sequence(player, shape, start, dir, block, capacity)
                case (0, False):
                    empty = True
                    if sub_len != 0:
                        shape.append(sub_len)
                        sub_len = 0
                case (p, _) if p == player:
                    sub_len += 1
                    start = current
                    empty = False
                case (p, _) if p != player:
                    if sub_len != 0:
                        shape.append(sub_len)
                    block = dir.get_block()
                    return Sequence(player, shape, start, dir, block, capacity)
            capacity += 1
            current += dir
        block = dir.get_block()
        if sub_len != 0:
            shape.append(sub_len)
        return Sequence(player, shape, start, dir, block, capacity)

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


# Shape possibilities:
# Seq len 2: (2), (1, 1)
# Seq len 3: (3), (2, 1), (1, 2), (1, 1, 1)
# Seq len 4: (4), (3, 1), (2, 2), (1, 3)
