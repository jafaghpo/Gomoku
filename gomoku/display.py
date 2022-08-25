from tkinter.tix import CELL
from tracemalloc import start
import pygame
import pygame_menu
import numpy as np
import sys
import time
from random import randint as rand
from copy import deepcopy
from gomoku.board import Board, Coord
from gomoku.engine import dumb_algo

CELL_SIZE = 40
BASE_SIZE = CELL_SIZE * 2
LINE_WIDTH = 2
PADDING = BASE_SIZE // 2

TEXT_PATH = "./assets/textures"


# Option menu inside the start menu
class OptionMenu:
    def __init__(self, display):
        self.menu = pygame_menu.Menu(
            "Options",
            display.screen_size,
            display.screen_size,
            theme=pygame_menu.themes.THEME_DARK,
        )
        self.display = display
        self.menu.add.dropselect(
            title="Pick a board size",
            items=[
                ("10", 10),
                ("11", 11),
                ("12", 12),
                ("13", 13),
                ("14", 14),
                ("15", 15),
                ("16", 16),
                ("17", 17),
                ("18", 18),
                ("19", 19),
                ("20", 20),
                ("21", 21),
                ("22", 22),
                ("23", 23),
                ("24", 24),
            ],
            font_size=20,
            default=9,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=5,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_board_size_change,
        )
        self.menu.add.text_input(
            "Algo time limit (ms): ",
            default="500",
            maxchar=4,
            onchange=self.on_time_change,
        )

        self.menu.add.button("Return to main menu", pygame_menu.events.RESET)

    def on_board_size_change(self, value: tuple, board_size: str):
        selected, index = value
        print(f'Selected difficulty: "{selected}" ({board_size}) at index {index}')
        self.display.args.size = int(board_size)
        self.display.screen_size = BASE_SIZE + (self.display.args.size - 1) * CELL_SIZE

    def on_time_change(self, time: str):
        selected = time
        print(f'Algo time limit (ms): "{selected}" ({time})')
        self.display.args.time = selected


# Pause menu that appears when the game is paused
class MatchMenu:
    def __init__(self, display):
        mytheme = pygame_menu.Theme(
            background_color=(0, 0, 0, 0),  # transparent background
            title_background_color=(4, 47, 126),
            title_font_shadow=True,
            widget_padding=25,
        )
        self.display = display
        self.menu = pygame_menu.Menu(
            "Pause",
            display.screen_size,
            display.screen_size,
            theme=mytheme,
        )

        self.menu.add.button("Resume", self.on_resume)
        self.menu.add.button("Quit", self.on_quit)

    def on_quit(self):
        sys.exit(pygame.quit())

    def on_resume(self):
        self.menu.close(self.display.run())


# Game menu that appears when the game is started, contains the option menu
class GameMenu:
    def __init__(self, display):
        self.menu = pygame_menu.Menu(
            "Gomoku",
            display.screen_size,
            display.screen_size,
            theme=pygame_menu.themes.THEME_DARK,
        )
        self.display = display
        self.player1_type = "human"
        self.player2_type = "engine"
        self.menu.add.button("Start", self.on_start)
        self.menu.add.selector(
            "Player 1",
            [("Human", "human"), ("Engine", "engine")],
            onchange=self.on_player1_change,
        )
        self.menu.add.selector(
            "Player 2",
            [("Engine", "engine"), ("Human", "human")],
            onchange=self.on_player2_change,
        )
        option_menu = OptionMenu(self.display)
        self.menu.add.button("Options", option_menu.menu)
        self.menu.add.button("Quit", self.on_quit)

    def on_player1_change(self, value: tuple, player: str):
        selected, index = value
        print(f'Selected difficulty: "{selected}" ({player}) at index {index}')
        self.player1_type = selected[1]

    def on_player2_change(self, value: tuple, player: str):
        selected, index = value
        print(f'Selected difficulty: "{selected}" ({player}) at index {index}')
        self.player2_type = selected[1]

    def on_start(self):
        self.display.args.players = [self.player1_type, self.player2_type]
        print("final args:")
        print(self.display.args)
        self.display.screen = pygame.display.set_mode(
            (self.display.screen_size, self.display.screen_size)
        )
        self.menu.close(self.display.run())

    def on_quit(self):
        sys.exit(pygame.quit())


class Display:
    def __init__(self, args):
        pygame.init()
        self.args = args
        print(self.args)
        self.screen_size = BASE_SIZE + (args.size - 1) * CELL_SIZE
        if self.args.connect4:
            print("wsh")
            self.args.players = ["human", "human"]
            self.args.size = 7
            self.background = pygame.image.load(f"{TEXT_PATH}/connect4_background.png")
            self.screen = pygame.display.set_mode((1920, 1200))
            pygame.display.set_caption("Connect4")
            self.stone_text = (
                f"{TEXT_PATH}/c4_red.png",
                f"{TEXT_PATH}/c4_yellow.png",
            )
        else:
            self.background = pygame.image.load(f"{TEXT_PATH}/classic_background.png")
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gomoku")
            self.stone_text = (
                f"{TEXT_PATH}/classic_black_stone.png",
                f"{TEXT_PATH}/classic_white_stone.png",
                f"{TEXT_PATH}/classic_help_stone.png",
            )

        self.board = None
        self.board_history = []
        self.player_turn = 0
        self.game_over = False
        self.match_menu = MatchMenu(self)

    def render_background(self) -> None:
        self.screen.blit(self.background, (0, 0))

    def render_connect4(self):
        self.g_x = 1920 / 6
        self.g_y = 1200 / 7
        for y in range(6):
            for x in range(7):
                stone = pygame.image.load(f"{TEXT_PATH}/c4_hole.png")
                stone = pygame.transform.scale(stone, (self.g_y, self.g_y))
                self.screen.blit(
                    stone,
                    (
                        x * self.g_x + (self.g_x / 2 - self.g_y / 2),
                        y * self.g_y + (self.g_y / 2),
                    ),
                )

    def render_board(self):
        if self.args.connect4:
            self.render_connect4()
        else:
            for y in range(self.args.size):
                pygame.draw.line(
                    self.screen,
                    [255, 255, 255],
                    (PADDING, y * CELL_SIZE + PADDING),
                    (
                        (self.args.size - 1) * CELL_SIZE + PADDING,
                        y * CELL_SIZE + PADDING,
                    ),
                    LINE_WIDTH,
                )
            for x in range(self.args.size):
                pygame.draw.line(
                    self.screen,
                    [255, 255, 255],
                    (x * CELL_SIZE + PADDING, PADDING),
                    (
                        x * CELL_SIZE + PADDING,
                        (self.args.size - 1) * CELL_SIZE + PADDING,
                    ),
                    LINE_WIDTH,
                )

    def update(self) -> None:
        pygame.display.update()

    def render_cell(self, pos: Coord, player: int) -> None:
        stone = pygame.image.load(self.stone_text[player])
        if self.args.connect4:
            stone = pygame.transform.scale(stone, (self.g_y, self.g_y))
            self.screen.blit(
                stone,
                (
                    (pos.x + 1) * self.g_x - self.g_x / 2 - self.g_y / 2,
                    (pos.y + 1) * self.g_y - self.g_y / 2,
                ),
            )
        else:
            stone = pygame.transform.scale(stone, (CELL_SIZE * 0.9, CELL_SIZE * 0.9))
            self.screen.blit(
                stone,
                (
                    pos.x * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
                    pos.y * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
                ),
            )

    def render_all_cells(self) -> None:
        indexes = np.argwhere(self.board.cells != 0)
        for index in indexes:
            y, x = index

            self.render_cell(Coord(y, x), self.board.cells[y][x] - 1)

    def render_last_move(self, pos: Coord) -> None:
        if self.args.connect4 == False:
            if self.board.last_move:
                self.render_cell(
                    self.board.last_move, self.board.cells[self.board.last_move] - 1
                )

            rect_size = CELL_SIZE // 6
            pygame.draw.rect(
                self.screen,
                [255, 0, 0],
                (
                    pos.x * CELL_SIZE + PADDING - rect_size // 2 + 1,
                    pos.y * CELL_SIZE + PADDING - rect_size // 2 + 1,
                    rect_size,
                    rect_size,
                ),
            )

    def render_help_move(self, pos: Coord) -> None:
        print(self.stone_text[2])
        stone = pygame.image.load(self.stone_text[2])
        stone = pygame.transform.scale(stone, (CELL_SIZE * 0.9, CELL_SIZE * 0.9))
        self.screen.blit(
            stone,
            (
                pos.x * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
                pos.y * CELL_SIZE + PADDING * 1.05 - CELL_SIZE // 2,
            ),
        )

    def get_valid_move(self) -> Coord | None:
        if self.args.connect4:
            pos = pygame.mouse.get_pos()
            print(pos)
            print(self.g_x, self.g_y)
            print((pos[0] - (self.g_x / 2) // 2) // self.g_x)
            x = int((pos[0] - (self.g_x / 2) // 2) // self.g_x)
            if self.board.can_place_c4(x):
                return self.board.get_pos_c4(x)
        else:
            pos = pygame.mouse.get_pos()
            x, y = ((p - PADDING // 2) // CELL_SIZE for p in pos)
            pos = Coord(y, x)
            if self.board.can_place(pos):
                return pos

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

    def handle_event(self) -> Coord | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(pygame.quit())
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                return self.get_valid_move()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(pygame.quit())
                if event.key == pygame.K_RETURN:
                    self.match_menu.menu.mainloop(self.screen)
                elif event.key == pygame.K_BACKSPACE:
                    self.cancel_last_moves()

    # Main loop of the whole game, runs until the game is over and after the start menu.
    def run(self) -> None:
        if not self.board:
            self.board = Board((self.args.size, self.args.size))
        self.render_background()
        self.render_board()
        self.render_all_cells()
        if self.board.last_move:
            self.render_last_move(self.board.last_move)
        self.update()
        player_type = self.args.players
        total_time = 0
        had_help = 0
        while True:
            time.sleep(0.01)
            pos = self.handle_event()
            if self.game_over:
                continue
            if player_type[self.player_turn] == "human":
                if self.args.helpmove:
                    if had_help == 0:
                        self.render_help_move(dumb_algo(self.board))
                        self.update()
                        had_help = 1
                if not pos:
                    continue
            else:
                had_help = 0
                start_time = time.time()
                pos = dumb_algo(self.board)
                end_time = time.time()
                total_time = end_time - start_time
                if not pos:
                    self.game_over = True
                    continue
            if total_time > 0:
                self.render_background()
                self.render_board()
                self.render_all_cells()
                font = pygame.font.SysFont(None, 24)
                img = font.render(f"{total_time:.4f}", True, (0, 0, 0))
                self.screen.blit(img, (10, 10))
            self.board_history.append(deepcopy(self.board))
            self.render_cell(pos, self.player_turn)
            self.render_last_move(pos)
            self.update()
            self.board.add_move(pos, self.player_turn + 1)

            self.player_turn ^= 1


# TODO:
# - set a min/max size for the board (the background texture doesn't fit after size 24)
# - fix the fact that every mouse button press creates a move
