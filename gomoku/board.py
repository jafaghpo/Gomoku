from dataclasses import dataclass
import numpy as np

from gomoku.sequence import Sequence, Coord, Block, DIRECTIONS


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
        for id in self.seq_map[pos].copy():
            if not id in self.seq_list:
                self.seq_map[pos].discard(id)
                continue
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
                    self.seq_list[seq.id].is_blocked = Block.HEAD
                    self.seq_map[seq.start - seq.direction].add(seq.id)
                elif pos == seq.end + seq.direction:
                    self.seq_list[seq.id].is_blocked = Block.TAIL
                    self.seq_map[seq.end + seq.direction].add(seq.id)
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
