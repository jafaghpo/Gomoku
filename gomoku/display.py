import pygame
import pygame_menu
import numpy as np
import sys
import time
from copy import deepcopy
from gomoku.board import Board, Coord
from gomoku.engine import Engine

SCREEN_SIZE = 1000
LINE_WIDTH = 2

LAST_MOVE_COLOR = (220, 20, 60)
SUGGESTED_MOVE_COLOR = (50, 205, 50)

TEXT_PATH = "./assets/textures"


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

    def render_indicator(self, pos: Coord, color: tuple[int, int, int]) -> None:
        """
        Render a red indicator on the current mouse position
        """
        rect_size = self.cell_size // 6
        pygame.draw.rect(
            self.screen,
            color,
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
        self.render_indicator(pos, LAST_MOVE_COLOR)

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
        self.board = Board(self.args)
        engine = Engine(self.args.time, self.args.depth)
        self.render_board(bg=True, grid=True, cells=True, last_move=True)
        suggestion = False
        while True:
            time.sleep(0.01)  # Reduces heavily the CPU usage
            pos = self.handle_event()
            if self.game_over:
                continue
            if self.args.players[self.player_turn] == "human":
                if self.args.move_suggestion and not suggestion:
                    suggestion, _ = engine.search_best_move(self.board)
                    self.render_indicator(suggestion.coord, SUGGESTED_MOVE_COLOR)
                    self.update()
                    suggestion = True
                if not pos or self.board.is_free_double(pos, self.player_turn):
                    continue
            else:
                move, engine_time = engine.search_best_move(self.board)
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
