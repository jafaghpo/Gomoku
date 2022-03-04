from argparse import Namespace
from gomoku.board import Board, Position
from gomoku.engine import dumb_algo
import numpy as np
import sys

import os

# Hide pygame prompt message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

OFFSET = 40
BASE_SIZE = OFFSET * 2
LINE_WIDTH = 2
PADDING = BASE_SIZE // 2


class Background(pygame.sprite.Sprite):
    def __init__(self, screen: pygame.Surface, background: str | list[int]):
        self.screen = screen
        if type(background) is tuple:
            self.bg_type = "color"
            screen.fill(background)
        else:
            self.bg_type = "image"
            pygame.sprite.Sprite.__init__(self)
            self.image = pygame.image.load(background)
            self.rect = self.image.get_rect()
            self.rect.left, self.rect.top = (0, 0)

    def update(self):
        if self.bg_type == "image":
            self.screen.blit(self.image, self.rect)


def is_valid_position(pos: Position, board: np.ndarray) -> bool:
    """
    Check if a position is valid
    """
    max_y, max_x = board.shape
    y, x = pos
    return 0 <= y < max_y and 0 <= x < max_x and board[pos] == 0


def pixel_to_board(pixel: Position, squares: bool) -> Position:
    """
    Convert a pixel coordinate to a board coordinate.
    Args:
        pixel: The pixel coordinates to convert.
        squares: Stone placement is done in squares instead of at intersections if True.
    """

    conditional_padding = PADDING if squares else PADDING // 2
    x, y = (((x - conditional_padding) // OFFSET) for x in pixel)
    return y, x


def exit_requested(event: pygame.event.Event) -> bool:
    """
    Check if the user wants to quit the game
    """
    return event.type == pygame.QUIT or (
        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    )


def handle_events() -> tuple[Position | None, bool]:
    """
    Handle user input
    Returns the position of the mouse click and whether the cancel button was pressed
    """
    for event in pygame.event.get():
        if exit_requested(event):
            sys.exit(pygame.quit())
        if event.type == pygame.MOUSEBUTTONDOWN:
            return tuple(pygame.mouse.get_pos()), False
        if event.type == pygame.KEYDOWN:
            # if key is backspace, cancel the move
            # if event.key == pygame.K_LEFT:
            #     return None, True
            if event.key == pygame.K_BACKSPACE:
                return None, True
    return None, False


def draw_board(board: Board, screen: pygame.Surface, args: Namespace) -> None:
    """
    Draw the board on the screen
    """

    squares = args.squares
    # Draw the grid
    max_y, max_x = (x + (1 if squares else 0) for x in board.shape)
    for y in range(max_y):
        pygame.draw.line(
            screen,
            [255, 255, 255],
            (PADDING, y * OFFSET + PADDING),
            ((max_x - 1) * OFFSET + PADDING, y * OFFSET + PADDING),
            LINE_WIDTH,
        )
    for x in range(max_x):
        pygame.draw.line(
            screen,
            [255, 255, 255],
            (x * OFFSET + PADDING, PADDING),
            (x * OFFSET + PADDING, (max_y - 1) * OFFSET + PADDING),
            LINE_WIDTH,
        )

    # Draw the stones


def draw_stone(
    board: Board,
    screen: pygame.Surface,
    args: Namespace,
    pos: Position,
) -> None:
    """
    Draw a stone on the screen
    """
    row, col = pos
    stone = pygame.image.load(args.stone_texture[board[row, col]])
    stone = pygame.transform.scale(stone, (OFFSET * 0.9, OFFSET * 0.9))
    screen.blit(
        stone,
        (
            col * OFFSET + PADDING * 1.05 - int(not args.squares) * OFFSET / 2,
            row * OFFSET + PADDING * 1.05 - int(not args.squares) * OFFSET / 2,
        ),
    )


def run(args: Namespace) -> None:
    """
    Run the game and display it on the screen
    """
    board = Board(args.rules.board)
    pygame.init()

    # Add space for an extra row and column if arg.display.squares is True
    screen = pygame.display.set_mode(
        tuple(
            (x - int(not args.display.squares)) * OFFSET + BASE_SIZE
            for x in board.shape[::-1]
        )
    )

    pygame.display.set_caption("Gomoku")
    is_game_running, should_render, current, player = True, False, 1, args.player
    board_history = [board.copy()]
    background = Background(screen, args.display.background)
    background.update()
    draw_board(board, screen, args.display)
    pygame.display.update()
    while True:
        pos, cancel_move = handle_events()
        if cancel_move and len(board_history) > 2:
            board_history = board_history[:-2]
            board = board_history[-1].copy()
            background.update()
            draw_board(board, screen, args.display)
            indexes = np.argwhere(board != 0)
            for index in indexes:
                draw_stone(board, screen, args.display, index)
            pygame.display.update()
            is_game_running = True

        if not is_game_running:
            continue
        if player[current] == "human":
            if not pos:
                continue
            position = pixel_to_board(pos, args.display.squares)
            if is_valid_position(position, board):
                board[position] = current
                draw_stone(board, screen, args.display, position)
                should_render = True
                current ^= 3
        else:
            position = dumb_algo(board)
            if position is None:
                is_game_running = False
            else:
                board[position] = current
                draw_stone(board, screen, args.display, position)
                should_render = True
                current ^= 3
        if should_render:
            board_history.append(np.copy(board))
            pygame.display.update()
            should_render = False
