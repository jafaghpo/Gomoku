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
        choices=range(3, 25),
        type=int,
        default=19,
        help="Size of the board",
    )
    parser.add_argument(
        "-c",
        "--capture",
        choices=range(0, 10),
        type=int,
        default=5,
        help="Number of captures to win. 0 for disabling capture",
    )
    parser.add_argument(
        "-w",
        "--win-sequence",
        choices=range(1, 10),
        type=int,
        default=5,
        help="Number of consecutive stones to win",
    )
    parser.add_argument(
        "-c4",
        "--connect4",
        action="store_true",
        help="Switch to connect4 gamemode instead of gomoku",
    )
    parser.add_argument(
        "-m",
        "--move-suggestion",
        action="store_true",
        help="Shows which move is the best for human players",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Prints debug information",
    )
    parser.add_argument(
        "-g",
        "--gravity",
        action="store_true",
        help="The stones will fall down at the bottom of the board like in connect4",
    )
    parser.add_argument(
        "-f",
        "--free-double",
        action="store_false",
        default=True,
        help="""Enable if specified the possibility to place a stone that introduces
        an unstoppable double free sequence scenario""",
    )

    args = parser.parse_args(argv)
    if args.win_sequence > args.size:
        parser.error("Board size must be greater than the winning sequence length")
    if args.free_double and args.win_sequence < 5:
        parser.error("Free double is not possible with a win sequence less than 5")
    return args


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
