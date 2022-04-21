import numpy as np
from gomoku.board import Board, Position
from time import time
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """
    A decorator that prints the execution time of the engine.
    """

    def wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Engine took {end - start:.4f} seconds")
        return result

    return wrapper


@timer
def dumb_algo(board: Board) -> Position | None:
    """
    A dumb algorithm that just returns the first available position.
    """
    pos = np.unravel_index(np.argmax(board.cells == 0, axis=None), board.cells.shape)
    print(f"Dumb algorithm chose {pos}")
    return tuple(map(int, pos)) if not board.cells[pos] else None
