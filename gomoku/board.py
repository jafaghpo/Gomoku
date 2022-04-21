from dataclasses import dataclass
from typing import NewType
import numpy as np

Position = NewType("Position", tuple[int, int])


@dataclass
class Sequence:
    """
    A sequence is a set of cells that are aligned in some way.
    """

    cells: list[Position]
    id: int
    player_id: int


@dataclass(init=False, repr=True)
class Board:
    """
    State of the board
    """

    cells: np.ndarray
    seq_map: dict[Position, list[int]]
    seq_list: dict[int, Sequence]

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
