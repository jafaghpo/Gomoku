from sys import argv
from argparse import ArgumentParser, Namespace
from gomoku.display import Display, GameMenu


def parse_args(argv: list[str]) -> Namespace:
    parser = ArgumentParser(prog="gomoku")
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=500,
        help="time limit for engine move in milliseconds",
    )
    parser.add_argument(
        "-p",
        "--players",
        nargs=2,
        choices={"human", "engine"},
        default=["human", "engine"],
        help="Player type (human or engine)",
        metavar=("player1", "player2"),
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=19,
        help="Size of the board",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(argv[1:])
    display = Display(args)
    if len(argv) == 1:
        game_menu = GameMenu(display)
        game_menu.menu.mainloop(display.screen)
    else:
        display.run()


if __name__ == "__main__":
    main()
