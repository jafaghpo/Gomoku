import numpy as np
from gomoku.board import Board, Coord
from time import time


def dumb_algo(board: Board) -> tuple[Coord | None, int]:
    """
    A dumb algorithm that just returns the first available position.
    """
    start = time()
    pos = np.unravel_index(np.argmax(board.cells == 0, axis=None), board.cells.shape)
    end = time()
    return (Coord(*map(int, pos)) if not board.cells[pos] else None, start - end)
