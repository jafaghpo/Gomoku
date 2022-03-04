from sys import exit
from argparse import Namespace
from gomoku.board import Board, Position
from gomoku.engine import dumb_algo


def get_valid_user_input(board: Board) -> Position:
    """
    Get user input and check if it is valid and return the position
    """
    invalid_input = True
    cancel_move = False
    pos = Position(0, 0)
    while invalid_input:
        user_input = input(
            'Enter a position (format: row column) or type "quit" to leave: '
        )
        if user_input == "quit":
            exit(0)
        elif user_input == "cancel":
            cancel_move = True
            break
        split_input = user_input.split()
        if len(split_input) != 2:
            print("Invalid input. " "Please enter a position in the format: row column")
            continue

        def is_valid_index(i, x):
            return x.isdigit() and int(x) in range(0, board.shape[i])

        if all(map(is_valid_index, range(len(board.shape)), split_input)):
            invalid_input = False
        else:
            print("Invalid input. " "Indexes are either not integers or out of range")
            continue
        pos = Position(*map(int, split_input))
        if board[pos] != 0:
            print("Invalid input. Position is not empty")
            invalid_input = True

    return pos, cancel_move


def run(args: Namespace) -> None:
    """
    Run the game and display it in the terminal
    """
    board = Board(args.rules.board)
    is_game_running = True
    current = 1  # player 1 starts
    board_history = [board.copy()]
    print(board, end="\n\n")
    while is_game_running:
        if args.player[current] == "human":
            position, cancel_move = get_valid_user_input(board)
            if cancel_move and len(board_history) > 2:
                board_history = board_history[:-2]
                board = board_history[-1].copy()
                print(f"Cancelled the last 2 moves.")
                print(board, end="\n\n")
                continue
            print(f"Player {current} (human) played at {position}")
            board[position] = current
            print(board, end="\n\n")
        else:
            position = dumb_algo(board)
            if position is None:
                print(f"Player {current} (engine) has no valid moves")
                is_game_running = False
            else:
                print(f"Player {current} (engine) played at {position}")
                board[position] = current
                print(board, end="\n\n")
        current ^= 3  # change player id from 1 to 2 and vice versa
        board_history.append(board.copy())
