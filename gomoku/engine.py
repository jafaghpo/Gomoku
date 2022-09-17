from gomoku.board import Board, Coord
from time import time
from dataclasses import dataclass, field


@dataclass
class Move:
    """
    A move is a node in the tree corresponding to a move played at a given coordinate
    on a state of the board. The impact of the move on the state of the board
    is stored in the score attribute during the tree evaluation
    """
    coord: Coord
    score: int = 0

@dataclass
class Successors:
    """
    Successors are the considered moves that can be played
    from a given state of the board
    - lst: list of moves/nodes from a given state of the board
    - best: the best move from a given state of the board
    - depth: the depth of the tree at which the successors are evaluated
    """
    lst: list[Move] = field(default_factory=list)
    best: Move | None = None
    depth: int = 0

@dataclass
class Tree:
    """
    A tree is a graph representing all possible game states.
    It is used to evaluate the best move to play.
    Each node is a move played on a state of the board.
    - nodes: a stack of all the successors, each successors being a list of moves
    - states: a stack of all the states of the board the tree is exploring.
        -> The first state is the current state of the board.
        -> When a move is played, the last state of the board is copied, updated
            and pushed to the stack.
        -> After the successors of a node are evaluated,
            the last state of the board is popped.
    - memory: a transposition table used to store the move order of
        the successors of a given state. The key is the hash of the board state and 
        the value is the list of considered moves from this state ordered by score.
    """
    nodes: list[Successors] = field(default_factory=list)
    states: list[Board] = field(default_factory=list)
    memory: dict[int, Successors] = field(default_factory=dict)
    
    def build(self, root: Board, depth: int) -> None:
        """
        Build the tree of possible by filling the nodes
        until the given depth is reached on the most left branch.

        ex: when depth = 3
                    root (node0)
                    /  \
                node1  node2    -> Succesors0 = [node1, node2], depth = 1
                /  \
            node3  node4        -> Succesors1 = [node3, node4], depth = 2
            /  \
        node5  node6            -> Succesors2 = [node5, node6], depth = 3
        
        nodes = [Succesors0, Succesors1, Succesors2]
        states = [root, root[node1], root[node1, node3], root[node1, node3, node5]
        
        Legend:
        - root: the current board
        - nodeX: a move
        - SuccesorsX: the list of possible moves from nodeX
        - root[node1, ..., nodeX]: the board state
            after playing the moves from node1 until move nodeX
        """
        self.states.append(root)
    
    def evaluate(self) -> Move:
        """
        Evaluate the built tree using a custom iterative alpha beta algorithm that uses
        the nodes & states stacks like a call stack in a classic recursive minimax algo
        in order to avoid recursion that would be too slow
        because of the size of the optimized implementation of board state.

        Go through the tree from the most left branch to the most right branch like in
        a recursive manner and evaluate the nodes at the top of the stack 
        when the depth is reached and pop the last state of the board and the last
        evaluated successors from the states & nodes stacks. Repeat until the stacks
        are empty.
        """
        pass
        

@dataclass
class Engine:
    """
    The engine is initialized with the following parameters:
    - max_depth: the maximum depth of the tree
    - time_limit: the time limit the engine can take to find the best move (in ms)
    - tree: the tree used to evaluate the best move
    """
    time_limit: int
    max_depth: int
    tree: Tree = field(default_factory=Tree)

    def time_elapsed(self, start: float) -> int:
        """
        Returns the time elapsed (in milliseconds) since the given start time
        """
        return round((time() - start) * 1000)

    def is_timeout(self, start: float) -> bool:
        """
        Returns whether the time limit has been reached
        """
        return self.time_elapsed(start) > self.time_limit - 100
    
    def search_best_move(self, root: Board) -> tuple[Move | None, int]:
        """
        Performs a minimax search with alpha-beta pruning and transposition table
        with iterative deepening and returns the best move and the time it took
        """
        start = time()
        best_move = None
        for depth in range(1, self.max_depth + 1):
            if self.is_timeout(start):
                break
            self.tree.build(root, depth)
            best_move = self.tree.evaluate()
        return best_move, self.time_elapsed(start)
