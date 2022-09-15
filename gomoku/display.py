from xmlrpc.client import FastUnmarshaller
import pygame
import pygame_menu
import numpy as np
import sys
import time
from copy import deepcopy
from gomoku.board import Board, Coord
from gomoku.engine import dumb_algo

SCREEN_SIZE = 1000
LINE_WIDTH = 2

TEXT_PATH = "./assets/textures"


class OptionMenu:
    """
    GUI to change the game options before starting a new game.
    Options are:
    - Board size
    - Time limit for the engine move
    """
    def __init__(self, display: "Display"):
        self.menu = pygame_menu.Menu(
            "Options",
            display.screen_size,
            display.screen_size,
            theme=pygame_menu.themes.THEME_DARK,
        )
        self.display = display
        self.menu.add.dropselect(
            title="Pick a board size",
            items=[(str(s), s) for s in range(3, 26)],
            font_size=20,
            default=19,
            open_middle=True,
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

    def on_board_size_change(self, _: tuple, board_size: str):
        """
        Callback for the board size dropselect.
        """
        self.display.args.board = int(board_size)
        self.display.screen_size = self.display.base_size + (self.display.args.board - 1) * self.display.cell_size

    def on_time_change(self, time: str):
        """
        Callback for the time limit text input.
        """
        if time.isdigit():
            self.display.args.time = int(time)


class MatchMenu:
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
            onchange=self.on_move_suggestion,
        )
        self.menu.add.button("Quit", self.on_quit)

    def on_quit(self):
        """
        Quit the game when the quit button is pressed
        """
        sys.exit(pygame.quit())

    def on_move_suggestion(self, value: tuple):
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



class GameMenu:
    """
    Main menu that appears when the game is launched.
    Options are:
    - Start a new game
    - Quit the game
    - Select the players type
    - Redirection to the options menu
    """
    def __init__(self, display: "Display"):
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

    def on_player1_change(self, value: tuple):
        """
        Change the player 1 type when the player 1 selector is changed
        """
        selected, _ = value
        self.player1_type = selected[1]

    def on_player2_change(self, value: tuple):
        """
        Change the player 2 type when the player 2 selector is changed
        """
        selected, _ = value
        self.player2_type = selected[1]

    def on_start(self):
        """
        Start a new game when the start button is pressed
        """
        self.display.args.players = {1: self.player1_type, -1: self.player2_type}
        self.display.screen = pygame.display.set_mode(
            (self.display.screen_size, self.display.screen_size)
        )
        self.menu.close(self.display.run())

    def on_quit(self):
        """
        Quit the game when the quit button is pressed
        """
        sys.exit(pygame.quit())


class Display:
    """
    Display the game and handle the user inputs
    """
    def __init__(self, args):
        pygame.init()
        self.args = args
        self.cell_size = SCREEN_SIZE // (self.args.board + 1)
        self.screen_size = self.cell_size * (self.args.board + 1)
        self.base_size = self.cell_size * 2
        self.screen_size = self.base_size + (args.board - 1) * self.cell_size
        self.padding = self.cell_size
        if args.connect4:
            self.background = pygame.image.load(f"{TEXT_PATH}/connect4_background.png")
        else:
            self.background = pygame.image.load(f"{TEXT_PATH}/classic_background.png")
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Gomoku")
        self.stone_text = {
            0: f"{TEXT_PATH}/classic_help_stone.png",
            1: f"{TEXT_PATH}/classic_black_stone.png",
            -1: f"{TEXT_PATH}/classic_white_stone.png",
            2: f"{TEXT_PATH}/c4_yellow.png",
            -2: f"{TEXT_PATH}/c4_red.png",
            3: f"{TEXT_PATH}/c4_hole.png",
        }
        self.board = None
        self.board_history = []
        self.player_turn = 1
        self.game_over = False
        self.match_menu = MatchMenu(self)

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
                (self.padding, i * self.cell_size + self.padding),
                (
                    (self.args.board - 1) * self.cell_size + self.padding,
                    i * self.cell_size + self.padding,
                ),
                LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                [255, 255, 255],
                (i * self.cell_size + self.padding, self.padding),
                (
                    i * self.cell_size + self.padding,
                    (self.args.board - 1) * self.cell_size + self.padding,
                ),
                LINE_WIDTH,
            )
    
    def render_holes(self):
        """
        Draw the holes of the game
        """
        for y in range(self.args.board):
            for x in range(self.args.board):
                self.render_cell(Coord(y, x), 3)

    def update(self) -> None:
        """
        Update the display
        """
        pygame.display.update()

    def render_cell(self, pos: Coord, player: int) -> None:
        """
        Render a cell on the board
        """
        index = 3
        if self.args.connect4:
            if player != 3:
                index = player * 2
        else:
            index = player
        stone = pygame.image.load(self.stone_text[index])
        stone = pygame.transform.scale(stone, (self.cell_size * 0.9, self.cell_size * 0.9))
        self.screen.blit(
            stone,
            (
                pos.x * self.cell_size + self.padding * 1.05 - self.cell_size // 2,
                pos.y * self.cell_size + self.padding * 1.05 - self.cell_size // 2,
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

    def render_last_move(self, pos: Coord) -> None:
        """
        Render a red indicator on the last move stone
        """
        if self.board.last_move:
            self.render_cell(
                self.board.last_move, self.board.cells[self.board.last_move]
            )
        rect_size = self.cell_size // 6
        pygame.draw.rect(
            self.screen,
            [255, 0, 0],
            (
                pos.x * self.cell_size + self.padding - rect_size // 2 + 1,
                pos.y * self.cell_size + self.padding - rect_size // 2 + 1,
                rect_size,
                rect_size,
            ),
        )

    def render_help_move(self, pos: Coord) -> None:
        """
        Render a transparent stone at the position suggested by the engine
        """
        pygame.draw.circle(
            self.screen,
            [124, 252, 0],
            (
                pos.x * self.cell_size + self.padding,
                pos.y * self.cell_size + self.padding,
            ),
            self.cell_size // 4,
        )
    
    def render_engine_time(self, time: float) -> None:
        """
        Render the time taken by the engine to compute the next move
        """
        self.render_board(bg=True, grid=True, cells=True, update=False)
        font = pygame.font.SysFont(None, 24)
        img = font.render(f"{time:.4f}", True, (0, 0, 0))
        self.screen.blit(img, (10, 10))
    
    def render_board(self, bg: bool = False, grid: bool = False, cells: bool = False, last_move: bool = False, update: bool = True) -> None:
        """
        Render the board
        """
        if bg:
            self.render_background()
        if grid:
            if self.args.connect4:
                self.render_holes()
            else:
                self.render_grid()
        if cells:
            self.render_all_cells()
        if self.board.last_move and last_move:
            self.render_last_move(self.board.last_move)
        if update:
            self.update()

    def get_valid_move(self) -> Coord | None:
        """
        Get the mouse position clicked by the user and return the corresponding
        cell position if it is a valid move
        """
        x, y = ((p - self.padding // 2) // self.cell_size for p in pygame.mouse.get_pos())
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
            self.board = self.board_history[0]
            self.board_history = []
            self.render_board(bg=True, grid=True)
            self.player_turn = 1
        else:
            self.board_history.pop()
            self.board = self.board_history.pop()
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
    
    def play_and_render_move(self, pos: Coord) -> None:
        """
        Play a move on the board
        """
        print(f"Move {max(len(self.board_history) // 2, 0)}: ", end='')
        self.board_history.append(deepcopy(self.board))
        self.render_cell(pos, self.player_turn)
        self.render_last_move(pos)
        captures = self.board.add_move(pos, self.player_turn)
        print(f"Player {self.player_turn if self.player_turn == 1 else 2} ", end='')
        print(f"placed a stone at {pos}")
        if  len(captures) > 0:
            p1, p2 = self.board.capture.values()
            print(f"Captured {len(captures)} enemy stones ({p1} to {p2}")
            self.render_board(bg=True, grid=True, cells=True, update=False)
            self.render_last_move(pos)
        self.update()

    def run(self) -> None:
        """
        Start the game loop
        """
        self.board = Board(self.args)
        self.render_board(bg=True, grid=True, cells=True, last_move=True)
        suggestion = False
        while True:
            time.sleep(0.01) # Reduces heavily the CPU usage
            pos = self.handle_event()
            if self.game_over:
                continue
            if self.args.players[self.player_turn] == "human":
                if self.args.move_suggestion and not suggestion:
                    self.render_help_move(dumb_algo(self.board)[0])
                    self.update()
                    suggestion = True
                if not pos or self.board.is_free_double(pos, self.player_turn):
                    continue
            else:
                suggestion = False
                pos, engine_time = dumb_algo(self.board)
                if not pos:
                    self.game_over = True
                    continue
                else:
                    self.render_engine_time(engine_time)
            self.play_and_render_move(pos)
            self.player_turn *= -1
            winner = self.board.is_game_over()
            if winner != 0:
                self.game_over = True
                print(f"Game over! {f'Player {winner} wins' if winner > 0 else 'Draw'}")
