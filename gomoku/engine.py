from gomoku.board import Board
from gomoku.coord import Coord
import time
from dataclasses import dataclass
import multiprocessing as mp

# Move = tuple[Coord, float]

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
    current_max_depth = 2
    debug: bool = False
    start_time: float = 0
    cutoff: int = 0
    evaluated_nodes: int = 0

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

    def time_elapsed(self) -> int:
        """
        Returns the time elapsed since the start of the search
        """
        return round(time.time() - self.start_time)
    
    def is_timeout(self) -> bool:
        """
        Returns True if the search has timed out
        """
        return self.time_elapsed() * 1000 >= self.time_limit - 50
    
    def first_move(self, root: Board) -> Move:
        """
        Returns the first move of the game
        """
        y, x = root.size // 2, root.size // 2
        if root.cells[y][x] == 0:
            return Move(root.get_valid_pos(y, x), root.score + root.cell_values[y][x])
        return Move(root.get_valid_pos(y - 1, x - 1), root.score + root.cell_values[y - 1][x - 1])
    

    def alpha_beta(self, state: Board, depth: int, alpha: float, beta: float) -> float:
        """
        Returns the best score for the engine by running an alpha-beta search
        """
        if depth == self.current_max_depth or state.is_game_over or self.is_timeout():
            self.evaluated_nodes += 1
            return state.score
        if state.playing == 1:
            print(f"Maximizing at depth {depth}")
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
            print(f"Minimizing at depth {depth}")
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


    def iterative_deepening(self, root: Board) -> tuple[float, int]:
        """
        Performs an iterative deepening search on the given state of the board
        using alpha-beta pruning
        """
        best = -float("inf") if root.playing == 1 else float("inf")
        for depth in range(2, self.max_depth + 1):
            if self.is_timeout():
                break
            self.current_max_depth = depth
            score = self.alpha_beta(root, depth, -float("inf"), float("inf"))
            print(f"Depth {depth} - Score: {score} - Time: {self.time_elapsed()}s - Evaluated nodes: {self.evaluated_nodes} - Cutoff: {self.cutoff}")
            best = max(best, score)
        return best, self.current_max_depth

    
    def search(self, root: Board) -> tuple[Move | None, int]:
        """
        Returns the best move for the engine by running an iterative deepening search
        """
        self.cutoff = self.evaluated_nodes = 0
        if len(root.stones) < 2:
            return self.first_move(root), time.time() / 1000
        print(f"Searching best move for player {1 if root.playing == 1 else 2}")
        self.start_time = time.time()
        results = []
        depth = 2
        for cell in root.successors:
            root.add_move(cell)
            score, depth_reached = self.iterative_deepening(root)
            print(f"Move {cell} reached depth {depth_reached} with score {score}")
            depth = max(depth, depth_reached)
            root.undo_last_move()
            results.append(Move(cell, score))
        results.sort(reverse=(root.playing == 1))
        if self.debug:
            self.debug_search(root, results, depth)
        return results[0], self.time_elapsed()
