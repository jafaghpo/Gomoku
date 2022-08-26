from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from collections import namedtuple

MAX_SEQ_LEN = 5


class Block(IntEnum):
    NO = 0
    HEAD = 1
    TAIL = 2
    BOTH = 3


# UNUSED
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

    def get_block_dir(self) -> Block:
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
    is_blocked: Block = Block.NO
    id: int = -1

    @property
    def stones(self) -> int:
        return sum(self.shape)
    
    @property
    def nb_holes(self) -> int:
        return len(self.shape) - 1

    @property
    def length(self) -> int:
        return self.stones + self.nb_holes

    @property
    def capacity(self) -> int:
        return self.length + sum(self.spaces)

    @property
    def end(self) -> Coord:
        return self.start + (self.length - 1) * self.direction

    @property
    def blocks(self) -> tuple[Coord]:
        if self.is_blocked == Block.NO:
            return ()
        elif self.is_blocked == Block.HEAD:
            return (self.start - self.direction,)
        elif self.is_blocked == Block.TAIL:
            return (self.end + self.direction,)
        else:
            return (self.start - self.direction, self.end + self.direction)

    @property
    def holes(self) -> tuple[Coord]:
        if self.nb_holes == 0:
            return ()
        holes = []
        acc = self.start
        for subseq_len in self.shape[:-1]:
            acc += self.direction * subseq_len
            holes.append(acc)
            acc += self.direction
        return tuple(holes)

    @property
    def growth_cells(self) -> tuple[tuple[Coord]]:
        head = tuple(
            self.start + (i + 1) * -self.direction
                for i in range(min(self.spaces[0], MAX_SEQ_LEN - self.length))
        )
        tail = tuple(self.end + (i + 1) * self.direction
            for i in range(min(self.spaces[1], MAX_SEQ_LEN - self.length)))
        return head, tail

    @property
    def rest_cells(self) -> tuple[Coord]:
        return self.start.range2d(self.direction, self.shape)

    @property
    def cost_cells(self) -> tuple[Coord]:
        cells = self.holes
        if self.is_blocked == Block.NO:
            return cells + (self.start - self.direction, self.end + self.direction)
        elif self.is_blocked == Block.HEAD:
            return cells + (self.end + self.direction,)
        elif self.is_blocked == Block.TAIL:
            return cells + (self.start - self.direction,)
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
            case (5, 0, _): return SeqType.FIVE
            case _ : return SeqType.DEAD

    def __str__(self):
        s = f"Sequence {self.id} (p{self.player}):\n"
        s += f"  stone count: {self.stones}\n"
        s += f"  length (including spaces): {self.length}\n"
        s += f"  shape: {self.shape}\n"
        s += f"  start: {self.start}\n"
        s += f"  direction: {DIR_STR[self.direction]}\n"
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
        s += f"direction={DIR_STR[self.direction]}, is_blocked={self.is_blocked}, id={self.id}, "
        s += f"spaces={self.spaces})"
        return s

    def __add__(self, other):
        self, other = (
            (self, other)
            if self.direction.get_block_dir() == Block.HEAD
            else (other, self)
        )
        self.shape[0] += other.shape[0] - 1
        self.shape = self.shape[::-1] + other.shape[1:]
        self.direction = other.direction
        self.is_blocked = Block(self.is_blocked + other.is_blocked)
        self.spaces = tuple(map(sum, zip(self.spaces, other.spaces)))
        return self


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

    def map_sequence_add(self, seq: Sequence) -> None:
        for cell in seq.rest_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.holes:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.cost_cells:
            self.seq_map.setdefault(cell, set()).add(seq.id)
        for side in seq.growth_cells:
            for cell in side:
                self.seq_map.setdefault(cell, set()).add(seq.id)
        for cell in seq.blocks:
            if cell[0] >= 0 and cell[1] >= 0:
                self.seq_map.setdefault(cell, set()).add(seq.id)
    
    def map_sequence_remove(self, seq: Sequence) -> None:
        for cell in seq.rest_cells:
            self.seq_map[cell].discard(seq.id)
        for cell in seq.holes:
            self.seq_map[cell].discard(seq.id)
        for cell in seq.cost_cells:
            self.seq_map[cell].discard(seq.id)
        for side in seq.growth_cells:
            for cell in side:
                self.seq_map[cell].discard(seq.id)
        for cell in seq.blocks:
            if cell[0] >= 0 and cell[1] >= 0:
                self.seq_map[cell].discard(seq.id)
    
    def find_capturable_sequences(self, pos: Coord, player: int) -> list[int]:
        if pos not in self.seq_map:
            return []
        capturable = []
        for id in self.seq_map[pos]:
            seq = self.seq_list[id]
            if player != seq.player and pos in seq.cost_cells and seq.shape == [2]:
                blocks = seq.blocks
                if len(blocks) == 1 and blocks[0].y >= 0 and blocks[0].x >= 0:
                    capturable.append(id)
        return capturable
            

    def add_move(self, pos: Coord, player: int) -> None:
        self.cells[pos] = player
        self.stones.add(pos)
        self.last_move = pos
        print(f"\nAdding move {pos}")
        capturable = self.find_capturable_sequences(pos, player)
        print(f"Capturable: {capturable}")
        checked_dir = set()
        to_remove = []
        for seq_id in self.seq_map.get(pos, set()).copy():
            seq = self.seq_list[seq_id]
            checked_dir.add(seq.direction)
            if seq.player == player or pos < seq.start - seq.direction or pos > seq.end + seq.direction:
                updated_seq = self.get_sequence(pos, seq.direction, player)
                if updated_seq is None:
                    continue
                updated_seq.id = seq.id
                self.seq_list[seq.id] = updated_seq
                self.map_sequence_add(updated_seq)
            else:
                if pos == seq.start - seq.direction:
                    self.seq_list[seq.id].is_blocked = Block(self.seq_list[seq.id].is_blocked + Block.HEAD)
                    for cell in seq.growth_cells[0]:
                        self.seq_map[cell].discard(seq.id)
                    self.seq_map[seq.start - seq.direction].add(seq.id)
                    self.seq_list[seq.id].spaces = (0, self.seq_list[seq.id].spaces[1])
                elif pos == seq.end + seq.direction:
                    self.seq_list[seq.id].is_blocked = Block(self.seq_list[seq.id].is_blocked + Block.TAIL)
                    for cell in seq.growth_cells[1]:
                        self.seq_map[cell].discard(seq.id)
                    self.seq_map[seq.end + seq.direction].add(seq.id)
                    self.seq_list[seq.id].spaces = (self.seq_list[seq.id].spaces[0], 0)
                elif seq.start < pos < seq.end:
                    seqs = tuple(filter(lambda s: s != None,
                        (self.get_sequence(seq.start, seq.direction, seq.player),
                        self.get_sequence(seq.end, seq.direction, seq.player))
                        ))
                    to_remove.append(seq)
                    for s in seqs:
                        s.id = self.last_seq_id
                        self.last_seq_id += 1
                        self.seq_list[s.id] = s
                        self.map_sequence_add(s)
            checked_dir.add(seq.direction)
            checked_dir.add(-seq.direction)
        for seq in to_remove:
            self.seq_list.pop(seq.id)
            self.map_sequence_remove(seq)
        for dir in DIRECTIONS:
            if dir in checked_dir:
                continue
            sequence = self.get_sequence(pos, dir, player)
            if not sequence:
                continue
            sequence.id = self.last_seq_id
            self.last_seq_id += 1
            self.seq_list[sequence.id] = sequence
            self.map_sequence_add(sequence)

        print("\nSequence map:")
        for cell, seq_list in self.seq_map.items():
            if seq_list:
                print(f"{cell}: {seq_list}")
        print(f"Last seq id: {self.last_seq_id}")
        print(f"Sequence list:")
        for seq in self.seq_list.values():
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
        is_blocked = Block.NO
        sub_len = 1
        spaces = [0, 0]
        block_dir = dir.get_block_dir()
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
                    return Sequence(
                        player, shape, start, dir, tuple(spaces), is_blocked
                    )
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
                case (p, False) if p != player:
                    spaces[idx] = 0
                    if sub_len != 0:
                        shape.append(sub_len)
                    is_blocked = block_dir
                    return Sequence(
                        player, shape, start, dir, tuple(spaces), is_blocked
                    )
                case (p, True) if p != player:
                    spaces[idx] = 1
                    if sub_len != 0:
                        shape.append(sub_len)
                    return Sequence(
                        player, shape, start, dir, tuple(spaces), is_blocked
                    )
            current += dir
        if not empty:
            is_blocked = block_dir
        if sub_len != 0:
            shape.append(sub_len)
        return Sequence(player, shape, start, dir, tuple(spaces), is_blocked)
    
    def get_sequence(self, current: Coord, dir: Coord, player: int) -> Sequence | None:
        head_seq = self.half_sequence(current, dir, player)
        tail_seq = self.half_sequence(current, -dir, player)
        sequence = head_seq + tail_seq
        if sequence.length >= 2:
            return sequence
        return None


# TODO:
# - Use cache for the Sequence properties that take non-negligeable time when repeated.
# - Fix sequences too long (ex: shape = [1,1,1,1])
# - Fix combining sequences with same direction
#   current behaviour is that both sequences have the same value
#   target behaviour is that the longer sequence is used and the shorter sequence is discarded
