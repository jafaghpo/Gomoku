from sys import argv
from argparse import ArgumentParser, Namespace
from gomoku.display import Display


def parse_args(argv: list[str]) -> Namespace:
    parser = ArgumentParser(prog="gomoku")
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=500,
        choices=range(100, 300100, 100),
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
        "-b",
        "--board",
        choices=range(3, 26),
        type=int,
        default=19,
        help="Size of the board",
    )
    parser.add_argument(
        "-c",
        "--capture-win",
        choices=range(0, 21),
        type=int,
        default=5,
        help="Number of captures to win. 0 for disabling capture",
    )
    parser.add_argument(
        "-s",
        "--sequence-win",
        choices=range(1, 10),
        type=int,
        default=5,
        help="Number of consecutive stones to win",
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
        an unstoppable double free sequence scenario
        (enabled for winning sequence less than 5)""",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=range(1, 11),
        default=10,
        help="Depth of the engine search",
    )

    args = parser.parse_args(argv)
    if args.sequence_win > args.board:
        print(f"Warning: Changed sequence win since it was greater than the board size")
        print(f"New sequence win value: {args.board}")
    if args.free_double and args.sequence_win < 5:
        args.free_double = False
        print("Warning: Free double is enabled due to winning sequence less than 5")
    return args


def main() -> None:
    args = parse_args(argv[1:])
    args.players = {1: args.players[0], -1: args.players[1]}
    display = Display(args)
    display.run()


if __name__ == "__main__":
    main()
