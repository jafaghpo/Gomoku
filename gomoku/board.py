from dataclasses import dataclass
import numpy as np
from functools import cache

from gomoku.sequence import Sequence, Coord, Block, MAX_SEQ_LEN


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
        self.cells = np.zeros(shape, dtype=int)
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
            if player != seq.player and pos in seq.cost_cells and seq.shape == (2,):
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
            checked_dir.add(seq.dir)
            if seq.player == player or pos < seq.start - seq.dir or pos > seq.end + seq.dir:
                updated_seq = self.get_sequence(pos, seq.dir, player)
                if updated_seq is None:
                    continue
                updated_seq.id = seq.id
                self.seq_list[seq.id] = updated_seq
                self.map_sequence_add(updated_seq)
            else:
                if pos == seq.start - seq.dir:
                    self.seq_list[seq.id].is_blocked = Block.HEAD
                    self.seq_map[seq.start - seq.dir].add(seq.id)
                elif pos == seq.end + seq.dir:
                    self.seq_list[seq.id].is_blocked = Block.TAIL
                    self.seq_map[seq.end + seq.dir].add(seq.id)
                elif seq.start < pos < seq.end:
                    seqs = tuple(filter(lambda s: s != None,
                        (self.get_sequence(seq.start, seq.dir, seq.player),
                        self.get_sequence(seq.end, seq.dir, seq.player))
                        ))
                    to_remove.append(seq)
                    for s in seqs:
                        s.id = self.last_seq_id
                        self.last_seq_id += 1
                        self.seq_list[s.id] = s
                        self.map_sequence_add(s)
            checked_dir.add(seq.dir)
            checked_dir.add(-seq.dir)
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

    @staticmethod
    @cache
    def slice_to_shape(slice: np.ndarray):
        print("Slicing")
        player = slice[0]
        shape = []
        sub_len = 1
        holed = False
        for i in range(1, len(slice)):
            if slice[i] == player:
                sub_len += 1
                holed = False
            else:
                if sub_len > 0:
                    shape.append(sub_len)
                    sub_len = 0
                if holed or slice[i] == player ^ 3:
                    return shape
                holed = True
        return shape if sub_len == 0 else shape + [sub_len]

    def slice_up(self, y: int, x: int) -> np.ndarray:
        return self.cells[max(y - 4, 0):y + 1, x][::-1]

    def slice_down(self, y: int, x: int) -> np.ndarray:
        return self.cells[y:y + MAX_SEQ_LEN, x]

    def slice_left(self, y: int, x: int) -> np.ndarray:
        return self.cells[y, max(x - 4, 0):x + 1][::-1]

    def slice_right(self, y: int, x: int) -> np.ndarray:
        return self.cells[y, x:x + MAX_SEQ_LEN]

    def slice_up_left(self, y: int, x: int) -> np.ndarray:
        y, x = self.cells.shape[0] - 1 - y, self.cells.shape[1] - 1 - x
        np.fliplr(np.flipud(self.cells))[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_up_right(self, y: int, x: int) -> np.ndarray:
        y = self.cells.shape[0] - 1 - y
        np.flipud(self.cells)[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_down_left(self, y: int, x: int) -> np.ndarray:
        x = self.cells.shape[1] - 1 - x
        np.fliplr(self.cells)[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_down_right(self, y: int, x: int) -> np.ndarray:
        return np.diagonal(self.cells[y:, x:])[:MAX_SEQ_LEN]
    
    def get_slice(self, pos: Coord, dir: Coord) -> np.ndarray:
        y, x = pos
        match dir:
            case (0, -1): return self.slice_left(y, x)
            case (0, 1): return self.slice_right(y, x)
            case (-1, 0): return self.slice_up(y, x)
            case (1, 0): return self.slice_down(y, x)
            case (-1, -1): return self.slice_up_left(y, x)
            case (-1, 1): return self.slice_up_right(y, x)
            case (1, -1): return self.slice_down_left(y, x)
            case (1, 1): return self.slice_down_right(y, x)
            case _: return None
    
    def get_spaces_after_seq(self, seq: Sequence) -> tuple[int, int]:
        current = seq.end
        count = 0
        while (current.in_range(self.cells.shape)
            and self.cells[current + seq.dir] == seq.player
            and self.cells[current + seq.dir] == 0):
            current += seq.dir
            count += 1
        return (count, 0) if seq.dir.get_block_dir() == Block.HEAD else (0, count)

    def get_half_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        print(f"Getting half sequence at {pos} in direction {dir}")
        board_slice = self.get_slice(pos, dir)
        print(board_slice)
        shape = self.slice_to_shape(tuple(board_slice))
        seq = Sequence(player, tuple(shape), pos, dir)
        seq.spaces = self.get_spaces_after_seq(seq)
        seq.is_blocked = Block.NO if sum(seq.spaces) == 0 else seq.dir.get_block_dir()
        return seq

    def get_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence | None:
        head_seq = self.get_half_sequence(pos, dir, player)
        tail_seq = self.get_half_sequence(pos, -dir, player)
        sequence = head_seq + tail_seq
        if sequence.length >= 2 and sequence.capacity >= 5:
            return sequence
        return None


# TODO:
# - Use cache for the Sequence properties that take non-negligeable time when repeated.
# - Fix sequences too long (ex: shape = [1,1,1,1])
