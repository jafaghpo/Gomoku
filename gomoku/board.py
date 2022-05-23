from dataclasses import dataclass
from typing import NewType
import numpy as np
from enum import IntEnum

Position = NewType("Position", tuple[int, int])


class ThreatType(IntEnum):
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

    type: ThreatType
    id: int
    player_id: int

    def __init__(self, cells: list[Position], id: int, player_id: int):
        self.cells = cells
        self.id = id
        self.player_id = player_id


@dataclass(init=False, repr=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Position, list[int]]
    seq_list: dict[int, Sequence]
    last_seq_id = 0
    last_move: Position | None = None

    def __init__(self, shape: tuple[int, int] = (19, 19)) -> None:
        self.cells = np.zeros(shape, dtype=np.uint8)
        self.sequence_map = {}
        self.sequences = {}

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
        # if not pos in self.seq_map:
        #     self.seq_map[pos] = [self.last_seq_id]
        #     self.seq_list[self.last_seq_id] = Sequence()
        #     self.last_seq_id += 1
