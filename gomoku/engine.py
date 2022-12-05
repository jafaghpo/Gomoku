from dataclasses import dataclass
from random import uniform, choice
import time

from gomoku.board import Board
from gomoku.coord import Coord

BEST_MOVES = 9

@dataclass
class Move:
    """
    A move is a node in the tree corresponding to a move played at a given coordinate
    on a state of the board. The impact of the move on the state of the board
    is stored in the score attribute during the tree evaluation
    """

    coord: Coord
    score: float | None = None

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self) -> str:
        return f"Move[coord={self.coord}, score={self.score}]"


@dataclass
class Engine:
    """
    Engine class
    """

    time_limit: int = 500
    max_depth: int = 10
    debug: bool = False
    start_time: float = 0
    difficulty: int = 0
    weight: float = 0.5
    current_max_depth: int = 2

    def debug_search(self, root: Board, moves: list[Move], depth: int) -> None:
        """
        Prints the successors of the root state of the board
        """
        print(f"Max depth {depth} reached")
        player_repr = {0: ".", 1: "\u001b[32mX\033[00m", -1: "\u001b[36mO\033[00m"}
        cells = [[player_repr[col] for col in row] for row in root.cells]
        y, x = root.move_history[-1][0]
        cells[y][x] = f"\033[91m{'X' if root.cells[y][x] == 1 else 'O'}\033[00m"
        for i, move in enumerate(moves[:9]):
            y, x = move.coord
            cells[y][x] = f"\u001b[33m{i + 1}\033[00m"
        print(" ".join([str(i % 10) for i in range(root.size)]))
        print("\n".join(" ".join(row + [str(i)]) for i, row in enumerate(cells)))
        for move in moves[:9]:
            print(f"{move.coord}: {move.score}")

    def optimize(func):
        """
        Decorator to optimize the alpha-beta search
        """

        def inner(self, root: Board) -> tuple[list[Move], int]:
            results = []
            best_depth = 0
            for cell in root.successors[:BEST_MOVES]:
                root.add_move(cell)
                score, depth_reached = func(self, root)
                root.undo_last_move()
                best_depth = max(best_depth, depth_reached)
                results.append(Move(cell, score))
            results.sort(reverse=(root.playing == 1))
            return results, best_depth

        return inner

    def change_weight(func):
        """
        Decorator to change the weight of the evaluation function
        """

        def inner(self, *args, **kwargs):
            calc_weight = self.time_elapsed
            res = func(self, *args, **kwargs)
            l, h = -self.weight * 0.02, self.weight * 0.02
            n = uniform(l, h)
            while calc_weight() < self.weight + n:
                l += 0.01
                h += 0.01
            return res

        return inner

    def time_elapsed(self) -> int:
        """
        Returns the time elapsed since the start of the search
        """
        return time.time() - self.start_time

    def is_timeout(self) -> bool:
        """
        Returns True if the search has timed out
        """
        return self.time_elapsed() >= self.time_limit - 0.01

    def first_move(self, root: Board) -> Move:
        """
        Returns the first move of the game
        """
        y, x = root.size // 2, root.size // 2
        if root.cells[y][x] == 0:
            return Move(root.get_valid_pos(y, x))
        return Move(root.get_valid_pos(y - 1, x - 1))

    def alpha_beta(self, state: Board, depth: int, alpha: float, beta: float) -> float:
        """
        Returns the best score for the engine by running an alpha-beta search
        """
        if self.is_timeout():
            raise TimeoutError
        if depth == self.current_max_depth or state.is_game_over():
            return state.score
        if state.playing == 1:
            score = -float("inf")
            for cell in state.successors:
                state.add_move(cell)
                score = max(score, self.alpha_beta(state, depth + 1, alpha, beta))
                state.undo_last_move()
                alpha = max(alpha, score)
                if alpha >= beta:
                    self.cutoff += 1
                    break
            return score
        else:
            score = float("inf")
            for cell in state.successors:
                state.add_move(cell)
                score = min(score, self.alpha_beta(state, depth + 1, alpha, beta))
                state.undo_last_move()
                beta = min(beta, score)
                if alpha >= beta:
                    self.cutoff += 1
                    break
            return score

    @change_weight
    @optimize
    def iterative_deepening(self, root: Board) -> tuple[float, int]:
        """
        Performs an iterative deepening search on the given state of the board
        using alpha-beta pruning
        """
        best = -float("inf") if root.playing == 1 else float("inf")
        for depth in range(2, self.max_depth + 1):
            self.current_max_depth = depth
            try:
                score = self.alpha_beta(root, depth, -float("inf"), float("inf"))
            except TimeoutError:
                break
            best = max(best, score)
        return best, self.current_max_depth

    def search(self, root: Board) -> tuple[Move | None, float]:
        """
        Returns the best move for the engine by running an iterative deepening search
        """
        self.start_time = time.time()
        if len(root.stones) < 2:
            return self.first_move(root), self.time_elapsed()
        results, depth = self.iterative_deepening(root)
        if self.debug:
            self.debug_search(root, results, depth)
        match self.difficulty:
            case 1:
                return choice(results[:3]), self.time_elapsed()
            case 2:
                return choice(results[:2]), self.time_elapsed()
            case _:
                return results[0], self.time_elapsed()
