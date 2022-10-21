import pygame
import pygame_menu
import numpy as np
import sys
import time
from copy import deepcopy
from gomoku.board import Board, Coord
from gomoku.engine import Engine

SCREEN_SIZE = 800
LINE_WIDTH = 2

LAST_MOVE_KEY = "last_move"
HELP_MOVE_KEY = "help_move"
LAST_MOVE_COLOR = (220, 20, 60)
HELP_MOVE_COLOR = (50, 205, 50)
INDICATOR = {
    LAST_MOVE_KEY: {"color": LAST_MOVE_COLOR, "ratio": 6},
    HELP_MOVE_KEY: {"color": HELP_MOVE_COLOR, "ratio": 3},
}


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
        self.menu.add.dropselect(
            title="Board size",
            dropselect_id="size",
            items=[(str(s), s) for s in range(3, 26)],
            font_size=20,
            default=16,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=20,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_board_size_change,
        )
        self.menu.add.dropselect(
            title="Captures to win",
            dropselect_id="capture",
            items=[("disabled", 0)] + [(str(s), s) for s in range(1, 21)],
            font_size=20,
            default=5,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=20,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_capture_win_change,
        )
        self.menu.add.dropselect(
            title="Winning sequence length",
            dropselect_id="sequence",
            items=[(str(s), s) for s in range(3, 10)],
            font_size=20,
            default=2,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=20,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_sequence_win_change,
        )
        self.menu.add.dropselect(
            title="Engine time limit (ms)",
            dropselect_id="time",
            items=[
                ("100", 100),
                ("250", 250),
                ("500", 500),
                ("750", 750),
                ("1000", 1000),
                ("2500", 2500),
                ("5000", 5000),
                ("7500", 7500),
                ("10000", 10000),
            ],
            font_size=20,
            default=2,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=17,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_time_change,
        )

        self.menu.add.selector(
            "Gravity",
            [
                ("Off", False),
                ("On", True),
            ],
            onchange=self.on_gravity_change,
        )

        self.menu.add.selector(
            "Free Double",
            [
                ("On", True),
                ("Off", False),
            ],
            onchange=self.on_freedouble_change,
        )

        self.menu.add.button("Return to main menu", pygame_menu.events.RESET)

    def on_board_size_change(self, value: tuple, board_size: str):
        """
        Function called to modify the board size from a selector in the option menu
        """
        self.change_board_size(int(board_size))
        if self.display.args.board < self.display.args.sequence_win:
            self.display.args.sequence_win = self.display.args.board
            self.menu.get_widget("sequence").set_value(self.display.args.board - 3)
        if self.display.args.board < 4:
            self.display.args.capture_win = 0
            self.menu.get_widget("capture").set_value(0)

    def on_time_change(self, value: tuple, time: str):
        """
        Function called to modify the time limit of the engine
        from a selector in the option menu
        """
        self.display.args.time = int(time)

    def on_capture_win_change(self, value: tuple, number: str):
        """
        Function called to modify the number of captures to win (1 capture = 2 stones)
        from a selector in the option menu
        """
        if self.display.args.board >= 4:
            self.display.args.capture_win = int(number)

    def on_sequence_win_change(self, value: tuple, number: str):
        """
        Function called to modify the number of stones in a row to win
        from a selector in the option menu
        """
        self.display.args.sequence_win = int(number)
        if self.display.args.sequence_win > self.display.args.board:
            self.change_board_size(self.display.args.sequence_win)
            self.menu.get_widget("size").set_value(self.display.args.board - 3)

    def change_board_size(self, size: int):
        """
        Change the board size and the screen and cell size
        """
        self.display.args.board = size
        self.display.cell_size = SCREEN_SIZE // (self.display.args.board + 1)
        self.display.screen_size = (
            self.display.cell_size * 2
            + (self.display.args.board - 1) * self.display.cell_size
        )

    def on_gravity_change(self, value: tuple, gravity: str):
        """
        Function called to modify the type of player 1 (human or engine)
        """
        selected, index = value
        self.display.args.gravity = selected[1]

    def on_freedouble_change(self, value: tuple, freedouble: str):
        """
        Function called to modify the type of player 1 (human or engine)
        """
        selected, index = value
        self.display.args.free_double = selected[1]


class GameMenu:
    """
    Game menu that appears when the game is started, contains the option menu
    Options are:
    - Access to Option Menu
    - Player type and order
    - Quit the game
    """

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
        self.menu.add.selector(
            "Game preset",
            [
                ("Classic", "classic"),
                ("Freestyle", "freestyle"),
                ("Connect4", "connect4"),
                ("Tic-tac-toe", "tictactoe"),
            ],
            onchange=self.on_gamemode_change,
        )
        option_menu = OptionMenu(self.display)
        self.menu.add.button("Options", option_menu.menu)
        self.menu.add.button("Quit", self.on_quit)

    def on_player1_change(self, value: tuple, player: str):
        """
        Function called to modify the type of player 1 (human or engine)
        """
        selected, index = value
        self.player1_type = selected[1]

    def on_player2_change(self, value: tuple, player: str):
        """
        Function called to modify the type of player 2 (human or engine)
        """
        selected, index = value
        self.player2_type = selected[1]

    def on_gamemode_change(self, value: tuple, gamemode: str):
        """
        Function called to quickly set a game mode.
        """
        selected, index = value
        gamemode = selected[1]
        match gamemode:
            case "connect4":
                self.display.args.gravity = True
                self.display.args.board = 7
                self.display.cell_size = SCREEN_SIZE // (self.display.args.board + 1)
                self.display.screen_size = (
                    self.display.cell_size * 2
                    + (self.display.args.board - 1) * self.display.cell_size
                )
                self.display.args.capture_win = 0
            case "tictactoe":
                self.display.args.board = 3
                self.display.cell_size = SCREEN_SIZE // (self.display.args.board + 1)
                self.display.screen_size = (
                    self.display.cell_size * 2
                    + (self.display.args.board - 1) * self.display.cell_size
                )
                self.display.args.sequence_win = 3
                self.display.args.capture_win = 0
                self.display.args.gravity = False

            case "freestyle":
                self.display.args.capture_win = 0
                self.display.args.free_double = False

    def on_start(self):
        """
        Function called when the start button is pressed to start the game
        """
        self.display.args.players = {1: self.player1_type, -1: self.player2_type}
        self.display.screen = pygame.display.set_mode(
            (self.display.screen_size, self.display.screen_size)
        )
        self.menu.close(self.display.run())

    def on_quit(self):
        """
        Function called when the quit button is pressed to quit the game
        """
        sys.exit(pygame.quit())


class PauseMenu:
    """
    Pause the game and display the pause menu.
    Options are:
    - Resume the game
    - Move suggestion
    - Quit the game
    """

    def __init__(self, display: "Display"):
        theme = pygame_menu.Theme(
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
            theme=theme,
        )

        self.menu.add.button("Resume", self.on_resume)
        self.menu.add.selector(
            "Move suggestion",
            [("Off", 0), ("On", 1)],
            default=int(display.args.move_suggestion),
            onchange=self.on_move_suggestion,
        )
        self.menu.add.button("Quit", self.on_quit)

    def on_quit(self):
        """
        Quit the game when the quit button is pressed
        """
        sys.exit(pygame.quit())

    def on_move_suggestion(self, value: tuple, _: int):
        """
        Enable or disable the move suggestion feature by selecting on or off
        """
        selected, _ = value
        self.display.args.move_suggestion = selected[1]

    def on_resume(self):
        """
        Resume the game when the resume button is pressed
        """
        self.menu.close(self.display.run())


class Display:
    """
    Display the game and handle the user inputs
    """

    def __init__(self, args):
        pygame.init()
        self.args = args
        print(self.args)
        self.cell_size = SCREEN_SIZE // (self.args.board + 1)
        self.screen_size = self.cell_size * 2 + (args.board - 1) * self.cell_size
        self.background = pygame.image.load(f"{TEXT_PATH}/classic_background.png")
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Gomoku")
        self.stone_text = {
            1: f"{TEXT_PATH}/classic_black_stone.png",
            -1: f"{TEXT_PATH}/classic_white_stone.png",
        }
        self.board = None
        self.engine = None
        self.board_history = []
        self.last_move = None
        self.player_turn = 1
        self.game_over = False
        self.match_menu = PauseMenu(self)

    def render_background(self) -> None:
        """
        Render the background of the game
        """
        self.screen.blit(self.background, (0, 0))

    def render_grid(self):
        """
        Draw the grid of the game
        """
        for i in range(self.args.board):
            pygame.draw.line(
                self.screen,
                [255, 255, 255],
                (self.cell_size, i * self.cell_size + self.cell_size),
                (
                    (self.args.board - 1) * self.cell_size + self.cell_size,
                    i * self.cell_size + self.cell_size,
                ),
                LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                [255, 255, 255],
                (i * self.cell_size + self.cell_size, self.cell_size),
                (
                    i * self.cell_size + self.cell_size,
                    (self.args.board - 1) * self.cell_size + self.cell_size,
                ),
                LINE_WIDTH,
            )

    def update(self) -> None:
        """
        Update the display
        """
        pygame.display.update()

    def render_cell(self, pos: Coord, player: int) -> None:
        """
        Render a cell on the board
        """
        stone = pygame.image.load(self.stone_text[player])
        stone = pygame.transform.scale(
            stone, (self.cell_size * 0.9, self.cell_size * 0.9)
        )
        self.screen.blit(
            stone,
            (
                pos.x * self.cell_size + self.cell_size * 1.05 - self.cell_size // 2,
                pos.y * self.cell_size + self.cell_size * 1.05 - self.cell_size // 2,
            ),
        )

    def render_all_cells(self) -> None:
        """
        Render all the cells on the board by getting all the non empty cells
        """
        indexes = np.argwhere(self.board.cells != 0)
        for index in indexes:
            y, x = index
            self.render_cell(Coord(y, x), self.board.cells[y][x])

    def render_indicator(self, pos: Coord, type: str) -> None:
        """
        Render a red indicator on the current mouse position
        """
        rect_size = self.cell_size // INDICATOR[type]["ratio"]
        pygame.draw.rect(
            self.screen,
            INDICATOR[type]["color"],
            (
                pos.x * self.cell_size + self.cell_size - rect_size // 2,
                pos.y * self.cell_size + self.cell_size - rect_size // 2,
                rect_size,
                rect_size,
            ),
        )

    def render_last_move(self, pos: Coord) -> None:
        """
        Render a red indicator on the last move stone
        """
        if self.last_move:
            self.render_cell(self.last_move, self.board.cells[self.last_move])
        self.render_indicator(pos, LAST_MOVE_KEY)

    def render_engine_time(self, time: float) -> None:
        """
        Render the time taken by the engine to compute the next move
        """
        self.render_board(bg=True, grid=True, cells=True, update=False)
        font = pygame.font.SysFont(None, 24)
        img = font.render(f"{time:.4f}", True, (0, 0, 0))
        self.screen.blit(img, (10, 10))

    def render_board(
        self,
        bg: bool = False,
        grid: bool = False,
        cells: bool = False,
        last_move: bool = False,
        update: bool = True,
    ) -> None:
        """
        Render the board
        """
        if bg:
            self.render_background()
        if grid:
            self.render_grid()
        if cells:
            self.render_all_cells()
        if self.last_move and last_move:
            self.render_last_move(self.last_move)
        if update:
            self.update()

    def get_valid_move(self) -> Coord | None:
        """
        Get the mouse position clicked by the user and return the corresponding
        cell position if it is a valid move
        """
        x, y = (
            (p - self.cell_size // 2) // self.cell_size for p in pygame.mouse.get_pos()
        )
        y = min(self.board.size - 1, max(0, y))
        x = min(self.board.size - 1, max(0, x))
        return self.board.get_valid_pos(y, x)

    def cancel_last_moves(self) -> None:
        """
        Cancel the last two moves played
        """
        if len(self.board_history) == 0:
            return
        if len(self.board_history) <= 2:
            self.board, self.last_move = self.board_history[0]
            self.board_history = []
            self.render_board(bg=True, grid=True)
            self.player_turn = 1
        else:
            self.board_history.pop()
            self.board, self.last_move = self.board_history.pop()
            self.render_board(bg=True, grid=True, cells=True, last_move=True)
        self.game_over = False

    def handle_event(self) -> Coord | None:
        """
        Handle user input
        """
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

    def play_and_render_move(self, move: Coord) -> None:
        """
        Play a move on the board
        """
        print(f"Move {max(len(self.board_history) // 2, 0)}: ", end="")
        self.board_history.append((deepcopy(self.board), self.last_move))
        self.render_cell(move, self.player_turn)
        self.render_last_move(move)
        captures = self.board.add_move(move, self.player_turn)
        self.last_move = move
        print(f"Player {self.player_turn if self.player_turn == 1 else 2} ", end="")
        print(f"placed a stone at {move}")
        if len(captures) > 0:
            p1, p2 = self.board.capture.values()
            print(f"Captured {len(captures)} enemy stones ({p1} to {p2})")
            self.render_board(bg=True, grid=True, cells=True, update=False)
            self.render_last_move(move)
        self.update()

    def run(self) -> None:
        """
        Start the game loop
        """
        if not self.board:
            self.board = Board(self.args)
            self.engine = Engine(self.args.time, self.args.depth)
        self.render_board(bg=True, grid=True, cells=True, last_move=True)
        suggestion = False
        while True:
            time.sleep(0.01)  # Reduces heavily the CPU usage
            pos = self.handle_event()
            if self.game_over:
                continue
            if self.args.players[self.player_turn] == "human":
                if self.args.move_suggestion and not suggestion:
                    suggestion, _ = self.engine.search_best_move(self.board)
                    self.render_indicator(suggestion.coord, HELP_MOVE_KEY)
                    self.update()
                    suggestion = True
                if not pos or self.board.is_free_double(pos, self.player_turn):
                    continue
            else:
                move, engine_time = self.engine.search_best_move(self.board)
                if not move:
                    self.game_over = True
                    continue
                else:
                    self.render_engine_time(engine_time)
                    pos = move.coord
            self.play_and_render_move(pos)
            self.player_turn *= -1
            suggestion = False
            winner = self.board.is_game_over()
            if winner != 0:
                self.game_over = True
                print(f"Game over! {f'Player {winner} wins' if winner > 0 else 'Draw'}")
