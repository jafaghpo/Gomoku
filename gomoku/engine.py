from audioop import reverse
from gomoku.board import Board, Coord
from time import time
from dataclasses import dataclass, field

BIG_NUM = int(1e20)

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

    def __init__(self, state: Board, depth: int):
        self.lst = [Move(coord, BIG_NUM * -state.playing) for coord in state.successors]
        self.depth = depth
        self.player = -state.playing
    
    def __iter__(self):
        return iter(self.lst)
    
    def __getitem__(self, index):
        return self.lst[index]
    
    def __len__(self):
        return len(self.lst)
    
    def __repr__(self):
        return f"Successors(depth={self.depth}, player={self.player}, lst={self.lst})"
    
    @property
    def best(self):
        return self[0]

    def filter(self) -> None:
        self.lst = [move for move in self.lst if move.score is not None]

    def sort(self) -> None:
        if self[0].score is None:
            return
        self.filter()
        self.lst.sort(reverse=self.player == -1)

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
    state_stack: list[Board] = field(default_factory=list)
    memory: dict[int, Successors] = field(default_factory=dict)
    start_time: float = 0

    @property
    def last_state(self):
        return self.state_stack[-1]

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
    
    def minimax(self, depth: int) -> int:
        """
        Minimax algorithm
        """
        if depth == 0 or self.last_state.is_game_over() or self.is_timeout():
            self.last_state.playing = -self.last_state.playing
            return self.last_state.score
        moves = None
        if hash(self.last_state) in self.memory:
            moves = self.memory[hash(self.last_state)]
        else:
            moves = Successors(self.last_state, depth)
        if self.last_state.playing == 1:
            value = -BIG_NUM
            for i in range(len(moves)):
                state = self.last_state.copy(moves[i].coord)
                self.state_stack.append(state)
                moves[i].score = max(value, self.minimax(depth - 1))
                self.state_stack.pop()
            moves.sort()
            self.memory[hash(self.last_state)] = moves
            return value
        else:
            value = BIG_NUM
            for i in range(len(moves)):
                state = self.last_state.copy(moves[i].coord)
                self.state_stack.append(state)
                moves[i].score = min(value, self.minimax(depth - 1))
                self.state_stack.pop()
            moves.sort()
            self.memory[hash(self.last_state)] = moves
            return value
    
    def maximize(self, moves: Successors, depth: int, alpha: int, beta: int) -> int:
        """
        Returns the best move for the maximizing player
        """
        value = -BIG_NUM
        for i in range(len(moves)):
            state = self.last_state.copy(moves[i].coord)
            self.state_stack.append(state)
            value = max(value, self.alpha_beta(depth - 1, alpha, beta))
            self.state_stack.pop()
            alpha = max(alpha, value)
            if value >= beta:
                break # Beta cut-off
            moves[i].score = value
        moves.sort()
        self.memory[hash(self.last_state)] = moves
        return value
    
    def minimize(self, moves: Successors, depth: int, alpha: int, beta: int) -> int:
        """
        Returns the best move for the minimizing player
        """
        value = BIG_NUM
        for i in range(len(moves)):
            state = self.last_state.copy(moves[i].coord)
            self.state_stack.append(state)
            value = min(value, self.alpha_beta(depth - 1, alpha, beta))
            self.state_stack.pop()
            beta = min(beta, value)
            if value <= alpha:
                break # Alpha cut-off
            moves[i].score = value
        moves.sort()
        if moves[0].score is not None:
            self.memory[hash(self.last_state)] = moves
        return value
    
    def alpha_beta(self, depth: int, alpha: int, beta: int) -> int:
        """
        Alpha-beta pruning algorithm
        """
        if depth == 0 or self.last_state.is_game_over() or self.is_timeout():
            self.last_state.playing *= -1
            return self.last_state.score
        moves = None
        if hash(self.last_state) in self.memory:
            moves = self.memory[hash(self.last_state)]
        else:
            moves = Successors(self.last_state, depth)
        if self.last_state.playing == 1:
            return self.maximize(moves, depth, alpha, beta)
        else:
            return self.minimize(moves, depth, alpha, beta)

    def MTDf_search(self, root: Board, depth: int, prev_best: Move | None) -> Move:
        """
        MTDf search algorithm
        """
        guess = prev_best.score if prev_best else root.score
        upper_bound = BIG_NUM
        lower_bound = -BIG_NUM
        self.state_stack.append(root)
        while lower_bound < upper_bound:
            beta = max(guess, lower_bound + 1)
            guess = self.alpha_beta(depth, beta - 1, beta)
            if guess < beta:
                upper_bound = guess
            else:
                lower_bound = guess
        return self.memory[hash(root)].best
    
    def first_move(self, root: Board) -> Move:
        """
        Returns the first move of the game
        """
        y, x = root.size // 2, root.size // 2
        if root.cells[y][x] == 0:
            return Move(Coord(y, x), root.score + root.cell_values[y][x])
        return Move(Coord(y - 1, x - 1), root.score + root.cell_values[y - 1][x - 1])

    def search_best_move(self, root: Board) -> tuple[Move | None, int]:
        """
        Uses an iterative deepening with MTDf search algorithm to find the best move
        """
        self.start_time = time()
        best_move = None
        if len(root.stones) < 2:
            return self.first_move(root), self.time_elapsed()
        for depth in range(1, self.max_depth + 1):
            if self.is_timeout():
                print("Time limit reached !")
                break
            best_move = self.MTDf_search(root, depth, best_move)
        return best_move, self.time_elapsed()