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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(argv[1:])
    display = Display()
    display.run(args)


if __name__ == "__main__":
    main()
