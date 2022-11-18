from gomoku.board import Board
from gomoku.coord import Coord
from time import time
from dataclasses import dataclass, field
import random

from copy import deepcopy

BIG_NUM = int(1e20)
BEST_MOVES = 7

@dataclass
class Move:
    """
    A move is a node in the tree corresponding to a move played at a given coordinate
    on a state of the board. The impact of the move on the state of the board
    is stored in the score attribute during the tree evaluation
    """
    coord: Coord
    score: int | None = None
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score
    
    def __repr__(self) -> str:
        return f"Move[coord={self.coord}, score={self.score}]"

@dataclass(init=False)
class Successors:
    """
    Successors are the considered moves that can be played
    from a given state of the board
    - lst: list of moves/nodes from a given state of the board
    - depth: the depth of the tree at which the successors are evaluated
    """
    lst: list[Move]
    depth: int
    player: int

    def __init__(self, state: Board, depth: int) -> None:
        lst = []
        for pos in state.successors:
            if not state.is_free_double(pos, state.playing):
                lst.append(Move(pos))
        self.lst = lst
        self.depth = len(state.move_history) + depth
        self.player = state.playing
    
    def __iter__(self):
        return iter(self.lst)
    
    def __getitem__(self, index):
        return self.lst[index]
    
    def __len__(self):
        return len(self.lst)
    
    def __repr__(self):
        return f"Successors(depth={self.depth}, player={self.player}, lst={self.lst})"
    
    def __contains__(self, coord: Coord) -> bool:
        return coord in [move.coord for move in self.lst]
    
    @property
    def best(self):
        return self[0]

    def filter(self) -> None:
        self.lst = [move for move in self.lst if move.score is not None]

    def sort(self) -> None:
        self.filter()
        self.lst.sort(reverse=(self.player == 1))
        self.lst = self.lst[:BEST_MOVES]

@dataclass
class Engine:
    """
    The engine is initialized with the following parameters:
    - max_depth: the maximum depth of the tree
    - time_limit: the time limit the engine can take to find the best move (in ms)
    - state_stack: a stack of all the states of the board the tree is exploring.
        -> The first state is the current state of the board.
        -> When a move is played, the last state of the board is copied, updated
            and pushed to the stack.
        -> After the successors of a node are evaluated,
            the last state of the board is popped.
    - memory: a transposition table used to store the move order of
        the successors of a given state. The key is the hash of the board state and 
        the value is the list of considered moves from this state ordered by score.
    """
    time_limit: int
    max_depth: int
    memory: dict[int, Successors] = field(default_factory=dict)
    start_time: float = 0
    current_max_depth: int = 0
    cutoff: int = 0
    memory_hits: int = 0
    evaluated_nodes: int = 0

    def debug_moves(self, root: Board, moves: Successors, depth: int) -> None:
        """
        Prints the successors of the root state of the board
        """
        print(f"\n\nDEPTH {depth} EVALUATION, TIME USED: {self.time_elapsed() / 1000:.2f}s")
        player_repr = {0: ".", 1: "\u001b[32mX\033[00m", -1: "\u001b[36mO\033[00m"}
        cells = [[player_repr[col] for col in row] for row in root.cells]
        y, x = root.move_history[-1][0]
        cells[y][x] = f"\033[91m{'X' if root.cells[y][x] == 1 else 'O'}\033[00m"
        for i, move in enumerate(moves.lst[:9]):
            cells[move.coord[0]][move.coord[1]] = f"\u001b[33m{i + 1}\033[00m"
        # for move in moves:
        #     print(f"{move.coord} -> {move.score}")
        print(" ".join([str(i % 10) for i in range(root.size)]))
        print("\n".join(" ".join(row + [str(i)]) for i, row in enumerate(cells)))
        print("\n")

    def time_elapsed(self) -> int:
        """
        Returns the time elapsed (in milliseconds) since the given start time
        """
        return round((time() - self.start_time) * 1000)

    def is_timeout(self) -> bool:
        """
        Returns whether the time limit has been reached
        """
        return self.time_elapsed() > self.time_limit - 100
    
    def clean_memory(self, depth: int) -> None:
        """
        Removes all the entries of the transposition table
        that are too old to be useful (based on the current depth)
        """
        for key, succ in self.memory.copy().items():
            if succ.depth < depth:
                del self.memory[key]
    
    def maximize(self, state: Board, moves: Successors, depth: int, alpha: int, beta: int) -> int:
        """
        Returns the best move for the maximizing player
        """
        # print(f"In maximize, depth={depth}, alpha={alpha}, beta={beta}")
        value = -BIG_NUM
        for i in range(len(moves)):
            state.add_move(moves[i].coord)
            score = self.alpha_beta(state, depth + 1, alpha, beta)
            value = max(value, score)
            state.undo_last_move()
            alpha = max(alpha, value)
            # print(f"value: {value}, score: {score}, coord: {moves[i].coord}, alpha: {alpha}, beta: {beta}")
            moves[i].score = score
            if value >= beta:
                self.cutoff += 1
                print(f"Beta cutoff")
                break # Beta cut-off
        moves.sort()
        if moves.lst:
            # print(f"after sort:")
            # self.debug_moves(state, moves)
            self.memory[hash(state)] = moves
        return value
    
    def minimize(self, state: Board, moves: Successors, depth: int, alpha: int, beta: int) -> int:
        """
        Returns the best move for the minimizing player
        """
        # print(f"In minimize, depth={depth}, alpha={alpha}, beta={beta}")
        value = BIG_NUM
        for i in range(len(moves)):
            state.add_move(moves[i].coord)
            score = self.alpha_beta(state, depth + 1, alpha, beta)
            value = min(value, score)
            state.undo_last_move() 
            beta = min(beta, value)
            # print(f"value: {value}, score: {score}, coord: {moves[i].coord}, alpha: {alpha}, beta: {beta}")
            moves[i].score = score
            if value <= alpha:
                self.cutoff += 1
                print(f"Alpha cut-off")
                break # Alpha cut-off
        moves.sort()
        if moves.lst:
            # print(f"after sort:")
            # self.debug_moves(state, moves)
            self.memory[hash(state)] = moves
        return value
    
    def alpha_beta(self, state: Board, depth: int, alpha: int, beta: int) -> int:
        """
        Alpha-beta pruning algorithm
        """
        if depth == self.current_max_depth or state.is_game_over() or self.is_timeout():
            print(f"playing: {state.playing}, last move: {state.cells[state.move_history[-1][0]]}({state.move_history[-1][0]})")
            score = state.score
            # print(f"depth: {depth}, move: {state.move_history[-1]}, score: {score}")
            self.evaluated_nodes += 1
            return score
        # print(f"depth: {depth}")
        moves = None
        if depth == 0 and hash(state) in self.memory:
            self.memory_hits += 1
            moves = self.memory[hash(state)]
        else:
            moves = Successors(state, depth)
        if state.playing == 1:
            return self.maximize(state, moves, depth, alpha, beta)
        else:
            return self.minimize(state, moves, depth, alpha, beta)

    def negamax_ab_pruning(self, state: Board, depth: int, alpha: int, beta: int) -> int:
        """
        Negamax with alpha-beta pruning
        """
        if depth == self.current_max_depth or state.is_game_over() or self.is_timeout():
            score = state.score
            self.evaluated_nodes += 1
            return score * state.playing
        # print(f"depth: {depth}")
        moves = None
        if depth == 0 and hash(state) in self.memory:
            self.memory_hits += 1
            moves = self.memory[hash(state)]
        else:
            moves = Successors(state, depth)
        value = -BIG_NUM
        for i in range(len(moves)):
            state.add_move(moves[i].coord)
            score = -self.negamax_ab_pruning(state, depth + 1, -beta, -alpha)
            value = max(value, score)
            state.undo_last_move()
            alpha = max(alpha, value)
            # print(f"value: {value}, score: {score}, coord: {moves[i].coord}, alpha: {alpha}, beta: {beta}")
            moves[i].score = score
            if value >= beta:
                self.cutoff += 1
                print(f"Beta cutoff")
                break

    def first_move(self, root: Board) -> Move:
        """
        Returns the first move of the game
        """
        y, x = root.size // 2, root.size // 2
        if root.cells[y][x] == 0:
            return Move(root.get_valid_pos(y, x), root.score + root.cell_values[y][x])
        return Move(root.get_valid_pos(y - 1, x - 1), root.score + root.cell_values[y - 1][x - 1])
    
    def quick_move(self, root: Board) -> Move:
        """
        Returns the best move with using a quick heuristic if the engine ran out of time
        to find a move
        """
        print(f"Using quick heuristic")
        cells = root.best_sequence_cost_cells
        if not cells:
            return Move(random.choice(list(root.successors)))
        return Move(random.choice(cells))

    def search_best_move(self, root: Board) -> tuple[Move | None, int]:
        """
        Uses an iterative deepening with MTDf search algorithm to find the best move
        """
        test = []
        print(f"time limit: {self.time_limit}")
        print(f"root board:\n{root}")
        self.start_time = time()
        self.memory_hits = self.cutoff = self.evaluated_nodes = 0
        best = None
        if len(root.stones) < 2:
            return self.first_move(root), self.time_elapsed() / 1000
        for depth in range(1, self.max_depth + 1):
            if self.is_timeout():
                print(f"Time limit reached for depth {depth}")
                break
            self.current_max_depth = depth
            self.clean_memory(len(root.move_history))
            self.alpha_beta(root, 0, -BIG_NUM, BIG_NUM)
            # self.negamax_ab_pruning(root, 0, -BIG_NUM, BIG_NUM)
            print("after negamax")
            if not hash(root) in self.memory:
                print(f"Time limit reached for depth {depth}")
                break
            print("before sort")
            self.debug_moves(root, self.memory[hash(root)], depth)
            best = self.memory[hash(root)].best
            # if depth == 1:
            #     test = deepcopy(self.memory[hash(root)])
            #     test.lst = test.lst[:5]
            # else:
            #     if not best.coord in test:
            #         print(f"Best move {best.coord} at depth {depth} not in the best moves at depth 1 {test}")
            #         sys.exit(1)
        print(f"Info: evaluated nodes={self.evaluated_nodes}, cutoffs={self.cutoff}, memory hits={self.memory_hits}")
        return best if best else self.quick_move(root), self.time_elapsed() / 1000