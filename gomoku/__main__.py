from sys import argv
from argparse import ArgumentParser, Namespace
from gomoku.display import Display, GameMenu, MatchMenu


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
        choices=range(10, 25),
        type=int,
        default=19,
        help="Size of the board",
    )
    parser.add_argument(
        "-c4",
        "--connect4",
        action="store_true",
        help="Switch to connect4 gamemode instead of gomoku",
    )
    parser.add_argument(
        "-hm",
        "--helpmove",
        action="store_true",
        help="Shows which move is the best for human players",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(argv[1:])
    args.players = {1: args.players[0], -1: args.players[1]}
    display = Display(args)
    if len(argv) == 1:
        game_menu = GameMenu(display)
        game_menu.menu.mainloop(display.screen)
    else:
        display.run()


if __name__ == "__main__":
    main()
