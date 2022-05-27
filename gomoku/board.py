from dataclasses import dataclass, field
from typing import NewType
import numpy as np
from enum import IntEnum

Position = NewType("Position", tuple[int, int])

NEIGHBORS_OFFSET = (
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
)

UP, UP_LEFT, UP_RIGHT = (-1, 0), (-1, -1), (-1, 1)
DOWN, DOWN_LEFT, DOWN_RIGHT = (1, 0), (1, -1), (1, 1)
LEFT, RIGHT = (0, -1), (0, 1)
DIRECTIONS = ((LEFT, RIGHT), (UP_LEFT, DOWN_RIGHT), (UP, DOWN), (UP_RIGHT, DOWN_LEFT))


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


@dataclass(init=False)
class Sequence:
    """
    A sequence is a set of cells that are aligned in some way.
    """

    type: SeqType | None = None
    id: int | None = None
    player: int | None = None
    dir: Position
    dist_start: int = 0
    max_len: int = 0
    hole: Position | None = None
    cells: list[Position]
    growth_cells: list[Position]
    block: list[Position]

    def __init__(self, dir: Position, player: int | None = None, id: int | None = None):
        self.cells = []
        self.growth_cells = []
        self.block = []
        self.id = id
        self.player = player
        self.dir = dir

    def __str__(self):
        return f"""Sequence {self.id}
- player: {self.player},
- direction: {self.dir},
- distance from start: {self.dist_start},
- maximum possible length: {self.max_len},
- hole position: {self.hole},
- stones in sequence: {self.cells},
- empty cells around sequence: {self.growth_cells},
- opponent stones blocking the sequence: {self.block})
"""


@dataclass(init=False, repr=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Position, list[int]]
    seq_list: dict[int, Sequence]
    children: set[Position]
    last_seq_id = 0
    last_move: Position | None = None

    def __init__(self, shape: tuple[int, int] = (19, 19)) -> None:
        print("Initializing board")
        self.cells = np.zeros(shape, dtype=np.uint8)
        self.sequence_map = {}
        self.sequences = {}
        self.children = set()

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "X", 2: "O"}
        return (
            "\n".join(map(lambda cell: player_repr[cell], row) for row in self.cells),
        )

    def valid_move(self, pos: Position) -> bool:
        return (
            0 <= pos[0] < self.cells.shape[0]
            and 0 <= pos[1] < self.cells.shape[1]
            and self.cells[pos] == 0
        )

    def add_move(self, pos: Position, player: int) -> None:
        self.cells[pos] = player
        self.last_move = pos
        self.children.update(self.generate_children(pos))
        self.children.discard(pos)
        # self.search_sequences(pos)
        # if not pos in self.seq_map:
        #    self.seq_map[pos] = [self.last_seq_id]
        #    self.seq_list[self.last_seq_id] = Sequence()
        #    self.last_seq_id += 1

    def generate_children(self, current: Position) -> list[Position]:
        def in_bounds_and_empty(pos: Position) -> bool:
            return (
                0 <= pos[0] < self.cells.shape[0]
                and 0 <= pos[1] < self.cells.shape[1]
                and self.cells[pos] == 0
            )

        def apply_offset(offset: Position) -> Position:
            return (current[0] + offset[0], current[1] + offset[1])

        neighbors = list(
            filter(
                in_bounds_and_empty,
                map(apply_offset, NEIGHBORS_OFFSET),
            )
        )
        return neighbors

    def within_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.cells.shape[0] and 0 <= pos[1] < self.cells.shape[1]

    def search_sequence(self, start: Position, dir: Position) -> Sequence | None:
        def increment_pos(pos: Position, dir: Position) -> None:
            return (pos[0] + dir[0], pos[1] + dir[1])

        def get_max_len(seq: Sequence, current: Position) -> int:
            max_len = seq.max_len
            while self.within_bounds(current) and self.cells[current] == 0:
                current = increment_pos(current, dir)
                max_len += 1
            return max_len

        seq = Sequence(dir)
        end_sequence = 0
        current = start

        while self.within_bounds(current) and end_sequence < 2:
            match (seq.player, self.cells[current]):
                # If no player was assigned to the sequence and the current cell is
                # empty, then increment the distance from the starting index
                case (None, 0):
                    seq.dist_start += 1

                # If the current cell contains a player id and no player
                # was assigned yet, then assign the player to the sequence
                # and increment the size of the sequence
                case (None, cell):
                    seq.player = cell
                    player = cell
                    seq.cells.append(current)

                # If the current cell is empty, check the next cell to see
                # if it's the sequence is holed or if this is the end of the sequence
                case (player, 0):
                    pos = increment_pos(current, dir)
                    match (
                        seq.hole,
                        self.within_bounds(pos) and self.cells[pos] == player,
                    ):
                        case (None, True):
                            seq.hole = current
                        case _:
                            end_sequence = True

                # If the current cell contains the same player id as the sequence,
                # increment the size of the sequence
                case (player, cell) if player == cell:
                    match not seq.hole or len(seq.cells) < 4:
                        case True:
                            seq.cells.append(current)
                        case False:
                            end_sequence = True

                # If there is an opponent stone, end the sequence
                case (player, cell) if player != cell:
                    seq.counter_cells.append(current)
                    end_sequence = True
            seq.max_len += 1
            current = increment_pos(current, dir)
        return seq if len(seq.cells) > 1 else None

    def search_sequences(self, pos: Position) -> list[Sequence]:
        sequences = []
        for dirs in DIRECTIONS:
            sequence = self.search_sequence(pos, dirs)
            if sequence:
                sequences.append(sequence)
        return sequences
