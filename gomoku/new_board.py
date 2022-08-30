from dataclasses import dataclass
import numpy as np
from functools import cache

from gomoku.sequence import MAX_SEQ_LEN, Sequence, Block, Coord, SeqType

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
    
    def add_stone(self, pos: Coord) -> None:
        self.stones.add(pos)
        self.cells[pos] = 1
        self.last_move = pos
    
    def update_sequences(self, seq: Sequence) -> None:
        self.seq_list[seq.id] = seq
        for coord in seq.coords:
            self.seq_map[coord].add(seq.id)

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
        np.fliplr(np.flipud(self))[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_up_right(self, y: int, x: int) -> np.ndarray:
        y = self.cells.shape[0] - 1 - y
        np.flipud(self)[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_down_left(self, y: int, x: int) -> np.ndarray:
        x = self.cells.shape[1] - 1 - x
        np.fliplr(self)[y:, x:].diagonal()[:MAX_SEQ_LEN]

    def slice_down_right(self, y: int, x: int) -> np.ndarray:
        return np.diagonal(self.cells[y:, x:])[:MAX_SEQ_LEN]
    
    def get_slice(self, pos: Coord, dir: Coord) -> np.ndarray:
        y, x = pos
        match dir:
            case (0, -1): return self.slice_left(self.cells, y, x)
            case (0, 1): return self.slice_right(self.cells, y, x)
            case (-1, 0): return self.slice_up(self.cells, y, x)
            case (1, 0): return self.slice_down(self.cells, y, x)
            case (-1, -1): return self.slice_up_left(self.cells, y, x)
            case (-1, 1): return self.slice_up_right(self.cells, y, x)
            case (1, -1): return self.slice_down_left(self.cells, y, x)
            case (1, 1): return self.slice_down_right(self.cells, y, x)
            case _: return None
    
    def get_spaces_after_seq(self, seq: Sequence) -> tuple[int, int]:
        current = seq.end
        while (self.can_place(current + seq.dir)
            or self.cells[current + seq.dir] == seq.player):
            current += seq.dir
            count += 1
        return (count, 0) if seq.dir.get_block_dir() == Block.HEAD else (0, count)

    def get_half_sequence(self, pos: Coord, dir: Coord, player: int) -> Sequence:
        board_slice = self.get_slice(pos, dir)
        shape = self.slice_to_shape(board_slice)
        seq = Sequence(player, tuple(shape), pos, dir)
        seq.spaces = self.get_spaces_after_seq(seq)
        seq.block = Block.NO if sum(seq.spaces) == 0 else seq.dir.get_block_dir()
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
