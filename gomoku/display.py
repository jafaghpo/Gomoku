import pygame
import pygame_menu
import numpy as np
import sys
from copy import deepcopy
from gomoku.board import Board, Position
from gomoku.engine import dumb_algo

CELL_SIZE = 40
BASE_SIZE = CELL_SIZE * 2
LINE_WIDTH = 2
PADDING = BASE_SIZE // 2

TEXT_PATH = "./assets/textures"


class OptionMenu:
    def __init__(self, display):
        self.menu = pygame_menu.Menu(
            "Options",
            display.screen_size,
            display.screen_size,
            theme=pygame_menu.themes.THEME_DARK,
        )
        self.display = display
        self.menu.add.text_input(
            "Board size :", default="19", maxchar=2, onchange=self.on_board_size_change
        )
        self.menu.add.text_input(
            "Game Time :", default="500", maxchar=4, onchange=self.on_time_change
        )

        self.menu.add.button("Return to main menu", pygame_menu.events.RESET)

    def on_board_size_change(self, board_size: str):
        selected = board_size
        print(f'Selected Board size: "{selected}" ({board_size})')
        if selected == "":
            selected = "9"
        self.display.args.size = int(selected)
        self.display.screen_size = BASE_SIZE + (self.display.args.size - 1) * CELL_SIZE

    def on_time_change(self, time: str):
        selected = time
        print(f'Selected Game time: "{selected}" ({time})')
        self.display.args.time = selected


class MatchMenu:
    def __init__(self, display):
        self.menu = pygame_menu.Menu(
            "Pause",
            display.screen_size,
            display.screen_size,
            theme=pygame_menu.themes.THEME_DARK,
        )
        self.display = display

        self.menu.add.button("Resume", self.menu.close)
        self.menu.add.button("Quit", self.on_quit)

    def on_quit(self):
        sys.exit(pygame.quit())


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
        self.screen_size = BASE_SIZE + (args.size - 1) * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Gomoku")

        self.background = pygame.image.load(f"{TEXT_PATH}/classic_background.png")

        self.stone_text = (
            f"{TEXT_PATH}/classic_black_stone.png",
            f"{TEXT_PATH}/classic_white_stone.png",
        )

        self.board = None
        self.board_history = []
        self.player_turn = 0
        self.game_over = False

    def render_background(self) -> None:
        self.screen.blit(self.background, (0, 0))

    def render_board(self):
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

    def handle_event(self, match_menu) -> Position | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(pygame.quit())
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                return self.get_valid_move()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit(pygame.quit())
                if event.key == pygame.K_ESCAPE:
                    match_menu.menu.mainloop(self.screen)
                elif event.key == pygame.K_BACKSPACE:
                    self.cancel_last_moves()

    def run(self) -> None:
        self.board = Board((self.args.size, self.args.size))
        self.render_background()
        self.render_board()
        self.update()
        match_menu = MatchMenu(self)
        player_type = self.args.players
        while True:
            pos = self.handle_event(match_menu)
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


# TODO:
# - set a min/max size for the board (the background texture doesn't fit after size 24)
# - fix the fact that every mouse button press creates a move
