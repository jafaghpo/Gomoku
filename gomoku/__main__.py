import gomoku.parser as parser
from sys import argv

if __name__ == '__main__':
    args_namespace = parser.parse_args(argv[1:])

    # debug print for the command line arguments
    args_dict = vars(args_namespace)
    for name in args_dict:
        print(f"{name}: {args_dict[name]}")
