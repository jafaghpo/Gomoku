import pygame
import pygame_menu
import numpy as np
import sys
from argparse import Namespace
from copy import deepcopy
from gomoku.board import Board, Position
from gomoku.engine import dumb_algo

BOARD_SIZE = 19
CELL_SIZE = 40
BASE_SIZE = CELL_SIZE * 2
LINE_WIDTH = 2
PADDING = BASE_SIZE // 2
SCREEN_SIZE = BASE_SIZE + (BOARD_SIZE - 1) * CELL_SIZE

TEXT_PATH = "./assets/textures"

class GameMenu:
    def __init__(self, args, display):
        self.menu = pygame_menu.Menu('Gomoku', SCREEN_SIZE, SCREEN_SIZE, theme=pygame_menu.themes.THEME_DARK)
        self.args = args
        self.display = display
        self.player1_type = "human"
        self.player2_type = "engine"
        self.menu.add.selector('Player 1',
                               [('Human', 'human'), ('Engine', 'engine')],
                               onchange=self.on_player1_change)
        self.menu.add.selector('Player 2',
                               [('Engine', 'engine'), ('Human', 'human')],
                               onchange=self.on_player2_change)
        self.menu.add.button('Start', self.on_start)
        self.menu.add.button('Quit', self.on_quit)


    def on_player1_change(self, value: tuple, player: str):
        selected, index = value
        print(f'Selected difficulty: "{selected}" ({player}) at index {index}')
        self.player1_type = selected[1]

    def on_player2_change(self, value: tuple, player: str):
        selected, index = value
        print(f'Selected difficulty: "{selected}" ({player}) at index {index}')
        self.player2_type = selected[1]

    def on_start(self):
        print("Allo ???")
        self.args.players = [self.player1_type, self.player2_type]
        print("final args:")
        print(self.args)
        self.menu.close(self.display.run(self.args))

    def on_quit(self):
        sys.exit(pygame.quit())
    


class Display:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Gomoku")

        self.background = pygame.image.load(f"{TEXT_PATH}/classic_background.png")
        self.render_background()
        self.render_board()
        self.update()

        self.stone_text = (
            f"{TEXT_PATH}/classic_black_stone.png",
            f"{TEXT_PATH}/classic_white_stone.png",
        )

        self.board = Board()
        self.board_history = []
        self.player_turn = 0
        self.game_over = False

    def render_background(self) -> None:
        self.screen.blit(self.background, (0, 0))

    def render_board(self):
        for y in range(BOARD_SIZE):
            pygame.draw.line(
                self.screen,
                [255, 255, 255],
                (PADDING, y * CELL_SIZE + PADDING),
                ((BOARD_SIZE - 1) * CELL_SIZE + PADDING, y * CELL_SIZE + PADDING),
                LINE_WIDTH,
            )
        for x in range(BOARD_SIZE):
            pygame.draw.line(
                self.screen,
                [255, 255, 255],
                (x * CELL_SIZE + PADDING, PADDING),
                (x * CELL_SIZE + PADDING, (BOARD_SIZE - 1) * CELL_SIZE + PADDING),
                LINE_WIDTH,
            )

    def update(self) -> None:
        pygame.display.update()

    def render_cell(self, pos: Position, player: int) -> None:
        stone = pygame.image.load(self.stone_text[player])
        stone = pygame.transform.scale(stone, (CELL_SIZE * 0.9, CELL_SIZE * 0.9))
        self.screen.blit(
            stone,
            (
                pos[1] * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
                pos[0] * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
            ),
        )

    def render_all_cells(self) -> None:
        indexes = np.argwhere(self.board.cells != 0)
        for index in indexes:
            index = tuple(index)
            self.render_cell(index, self.board.cells[index] - 1)

    def render_last_move(self, pos: Position) -> None:
        if self.board.last_move:
            self.render_cell(
                self.board.last_move, self.board.cells[self.board.last_move] - 1
            )

        rect_size = CELL_SIZE // 6
        pygame.draw.rect(
            self.screen,
            [255, 0, 0],
            (
                pos[1] * CELL_SIZE + PADDING - rect_size // 2 + 1,
                pos[0] * CELL_SIZE + PADDING - rect_size // 2 + 1,
                rect_size,
                rect_size,
            ),
        )

    def get_valid_move(self) -> Position | None:
        pos = pygame.mouse.get_pos()
        x, y = ((p - PADDING // 2) // CELL_SIZE for p in pos)
        if self.board.valid_move((y, x)):
            return (y, x)

    def cancel_last_moves(self) -> None:
        if len(self.board_history) <= 2:
            self.board = Board()
            self.board_history = []
            self.render_background()
            self.render_board()
            self.player_turn = 0
        else:
            self.board_history.pop()
            self.board = self.board_history.pop()
            self.render_background()
            self.render_board()
            self.render_all_cells()
            self.render_last_move(self.board.last_move)
        self.update()
        self.game_over = False

    def handle_event(self) -> Position | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(pygame.quit())
            if event.type == pygame.MOUSEBUTTONDOWN:
                return self.get_valid_move()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit(pygame.quit())
                if event.key == pygame.K_ESCAPE:
                    sys.exit(pygame.quit())
                elif event.key == pygame.K_BACKSPACE:
                    self.cancel_last_moves()

    def run(self, args: Namespace) -> None:
        self.render_background()
        self.render_board()
        self.update()
        player_type = args.players
        while True:
            pos = self.handle_event()
            if self.game_over:
                continue
            if player_type[self.player_turn] == "human":
                if not pos:
                    continue
            else:
                pos = dumb_algo(self.board)
                if not pos:
                    self.game_over = True
                    continue
            self.board_history.append(deepcopy(self.board))
            self.render_cell(pos, self.player_turn)
            self.render_last_move(pos)
            self.update()
            self.board.add_move(pos, self.player_turn + 1)
            self.player_turn ^= 1
