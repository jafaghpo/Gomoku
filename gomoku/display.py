import pygame
import pygame_menu
import numpy as np
import sys
import time
from copy import deepcopy
from gomoku.board import Board, Coord, GameOver, Sequence
from gomoku.engine import Engine
import gomoku.coord as coord

import cProfile
import pstats

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
        self.menu.add.dropselect(
            title="Engine depth limit",
            dropselect_id="depth",
            items=[
                ("1", 1),
                ("2", 2),
                ("3", 3),
                ("4", 4),
                ("5", 5),
                ("6", 6),
                ("7", 7),
                ("8", 8),
                ("9", 9),
                ("10", 10),
            ],
            font_size=20,
            default=9,
            open_middle=True,  # Opens in the middle of the menu
            selection_box_height=17,
            selection_box_width=212,
            selection_infinite=True,
            selection_option_font_size=20,
            onchange=self.on_depth_change,
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

    def on_depth_change(self, value: tuple, depth: str):
        """
        Function called to modify the depth limit of the engine
        from a selector in the option menu
        """
        self.display.args.depth = int(depth)

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
            "Difficulty",
            [
                ("Default", "Default"),
                ("Easy", "Easy"),
                ("Normal", "Normal"),
                ("Hard", "Hard"),
            ],
            onchange=self.on_difficulty_change,
        )
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
        self.display.args.players = {1: selected[1], -1: self.display.args.players[-1]}

    def on_player2_change(self, value: tuple, player: str):
        """
        Function called to modify the type of player 2 (human or engine)
        """
        selected, index = value
        self.display.args.players = {1: self.display.args.players[1], -1: selected[1]}

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

    def on_difficulty_change(self, value: tuple, difficulty: str):
        """
        Function called to quickly set a game difficulty.
        """
        selected, index = value
        difficulty = selected[1]
        match difficulty:
            case "Default":
                self.display.args.players = {1: "human", -1: "engine"}
                self.display.args.time = 500
                self.display.args.depth = 10
            case "Easy":
                self.display.args.players = {1: "human", -1: "engine"}
                self.display.args.time = 100
                self.display.args.depth = 1
            case "Normal":
                self.display.args.players = {1: "human", -1: "engine"}
                self.display.args.time = 750
                self.display.args.depth = 3

            case "Hard":
                self.display.args.players = {1: "engine", -1: "human"}
                self.display.args.time = 2000
                self.display.args.depth = 10

    def on_start(self):
        """
        Function called when the start button is pressed to start the game
        """
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
        self.menu.add.button("Restart", self.on_restart)
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
    
    def on_restart(self):
        """
        Restart the game when the restart button is pressed
        """
        self.display.board.reset()
        self.display.board_history = []
        self.display.last_move = None
        self.display.player_turn = 1
        self.display.game_over = False
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
                pos[1] * self.cell_size + self.cell_size * 1.05 - self.cell_size // 2,
                pos[0] * self.cell_size + self.cell_size * 1.05 - self.cell_size // 2,
            ),
        )

    def render_all_cells(self) -> None:
        """
        Render all the cells on the board by getting all the non empty cells
        """
        indexes = np.argwhere(self.board.cells != 0)
        for index in indexes:
            y, x = index
            self.render_cell((y, x), self.board.cells[y][x])

    def render_indicator(self, pos: Coord, type: str) -> None:
        """
        Render a red indicator on the current mouse position
        """
        rect_size = self.cell_size // INDICATOR[type]["ratio"]
        pygame.draw.rect(
            self.screen,
            INDICATOR[type]["color"],
            (
                pos[1] * self.cell_size + self.cell_size - rect_size // 2,
                pos[0] * self.cell_size + self.cell_size - rect_size // 2,
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
        font = pygame.font.SysFont(None, 24)
        img = font.render(f"{time:.4f}", True, (0, 0, 0))
        self.screen.blit(img, (10, 10))

    def render_number_captures(self) -> None:
        """
        Render the number of captures for each player.
        """
        font = pygame.font.SysFont(None, 24)
        p1, p2 = self.board.capture.values()
        img = font.render(f"Captures:   {p1} : {p2}", True, (0, 0, 0))
        self.screen.blit(img, (SCREEN_SIZE - 150, 10))

    def render_player_win(self, msg: str) -> None:
        """
        Render the annoncement of player winning.
        """
        font = pygame.font.SysFont(None, 24)
        img = font.render(msg, True, (0, 0, 0))
        self.screen.blit(img, (SCREEN_SIZE - (SCREEN_SIZE / 2 + 100 + len(msg)), 10))

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

    def play_and_render_move(self, move: Coord, engine_time: int) -> None:
        """
        Play a move on the board
        """
        print(f"Turn {max(len(self.board_history) // 2 + 1, 0)}: ", end="")
        self.board_history.append((deepcopy(self.board), self.last_move))
        self.render_cell(move, self.player_turn)
        self.render_last_move(move)
        captures = self.board.add_move(move)
        ###### DEBUG ######
        flag = False
        # print(f"Board before undo: {self.board}")
        # print(f"Board successors: {self.board.successors}")
        # self.board.undo_last_move()
        # print(f"Board after undo: {self.board}")
        # captures = self.board.add_move(move)
        # print(f"Board after move: {self.board}")
        for seq in self.board.seq_list.values():
            for stone in seq:
                if self.board.cells[stone] != seq.player:
                    print(f"Error: invalid rest cell {stone} in sequence {seq.id}")
                    flag = True
            for stone in seq.block_cells:
                if (
                    coord.in_bound(stone, self.board.size)
                    and self.board.cells[stone] != -seq.player
                ):
                    print(f"Error: invalid block cell {stone} in sequence {seq.id}")
                    flag = True
        if flag:
            sys.exit(pygame.quit())
        ###### END DEBUG ######
        self.last_move = move
        print(f"{'Black' if self.player_turn == 1 else 'White'} ", end="")
        print(f"placed a stone at {move}", end="")
        if len(captures) > 0:
            p1, p2 = self.board.capture.values()
            print(f" and captured {len(captures)} enemy stones")
            print(f"Captures count => Black: {p1} | White: {p2}")
            self.render_board(bg=True, grid=True, cells=True, update=False)
            self.render_last_move(move)
        else:
            print()
        if Board.debug:
            print(self.board)
        self.render_board(bg=True, grid=True, cells=True, update=False)
        if self.args.capture_win:
            self.render_number_captures()
        if engine_time:
            self.render_engine_time(engine_time)
        self.render_last_move(move)
        self.update()
    
    def game_over_to_string(self, game_over: GameOver) -> str:
        """
        Return the game over message
        """
        match game_over:
            case GameOver.DRAW:
                return "The game is a draw"
            case GameOver.BLACK_SEQUENCE_WIN:
                return "Black won by forming a winning sequence"
            case GameOver.WHITE_SEQUENCE_WIN:
                return "White won by forming a winning sequence"
            case GameOver.BLACK_CAPTURE_WIN:
                return "Black won by capture count"
            case GameOver.WHITE_CAPTURE_WIN:
                return "White won by capture count"
    
    def modify_capture_weight(self, status: GameOver) -> None:
        """
        Modify the capture weight if the game is over
        """
        match status:
            case GameOver.BLACK_SEQUENCE_WIN | GameOver.WHITE_SEQUENCE_WIN:
                Sequence.capture_weight *= 0.9
            case GameOver.BLACK_CAPTURE_WIN | GameOver.WHITE_CAPTURE_WIN:
                Sequence.capture_weight *= 1.1
            case _:
                pass

    def run(self) -> None:
        """
        Start the game loop
        """
        if not self.board:
            self.board = Board(self.args)
            self.engine = Engine(self.args.time, self.args.depth, self.args.debug)
        self.render_board(bg=True, grid=True, cells=True, last_move=True)
        suggestion = False
        engine_time = None
        print(self.args)
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
                with cProfile.Profile() as pr:
                    move, engine_time = self.engine.search_best_move(
                        deepcopy(self.board)
                    )
                # stats = pstats.Stats(pr)
                # stats.sort_stats(pstats.SortKey.TIME)
                # if len(self.board_history) > 2:
                #     stats.dump_stats(
                #         f"gomoku_b{self.args.board}_d{self.args.depth}_t{self.args.time}.prof"
                #     )
                if not move:
                    self.game_over = True
                    continue
                else:
                    pos = move.coord
            self.play_and_render_move(pos, engine_time)
            self.player_turn *= -1
            suggestion = False
            status = self.board.is_game_over()
            if status != GameOver.NONE:
                self.game_over = True
                self.modify_capture_weight(status)
                game_over_msg = self.game_over_to_string(status)
                self.render_player_win(game_over_msg)
                self.update()
