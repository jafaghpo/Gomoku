import gomoku.parser as parser
from sys import argv
import gomoku.cli as cli
import gomoku.gui as gui


def main() -> None:
    args = parser.parse_args(argv[1:])

    ### DEBUG
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"{k}: {v}")
    ### END DEBUG

    match args.display.user_interface:
        case "cli":
            cli.run(args)
        case "gui":
            gui.run(args)


if __name__ == "__main__":
    main()
