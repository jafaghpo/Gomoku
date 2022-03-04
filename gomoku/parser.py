from argparse import ArgumentParser, Namespace

GAME_MODE_DEFAULT_RULES = {
    "42": {
        "board": [19, 19],
        "winning_sequence": 5,
        "overline": None,
        "free_double": "both",
        "capture": 5,
        "gravity": False,
    },
    "freestyle": {
        "board": [15, 15],
        "winning_sequence": 5,
        "overline": None,
        "free_double": None,
        "capture": None,
        "gravity": False,
    },
    "renju": {
        "board": [15, 15],
        "winning_sequence": 5,
        "overline": "p1",
        "free_double": "p1",
        "capture": None,
        "gravity": False,
    },
    "omok": {
        "board": [19, 19],
        "winning_sequence": 5,
        "overline": "p1",
        "free_double": "p1",
        "capture": None,
        "gravity": False,
    },
    "ninuki_renju": {
        "board": [15, 15],
        "winning_sequence": 5,
        "overline": "p1",
        "free_double": "p1",
        "capture": 5,
        "gravity": False,
    },
    "pente": {
        "board": [19, 19],
        "winning_sequence": 5,
        "overline": None,
        "free_double": None,
        "capture": 5,
        "gravity": False,
    },
    "connect4": {
        "board": [6, 7],
        "winning_sequence": 4,
        "overline": None,
        "free_double": None,
        "capture": None,
        "gravity": True,
    },
    "tictactoe": {
        "board": [3, 3],
        "winning_sequence": 3,
        "overline": None,
        "free_double": None,
        "capture": None,
        "gravity": False,
    },
}


GAME_MODE_DEFAULT_THEME = {
    "42": {},
    "freestyle": {},
    "renju": {},
    "omok": {},
    "ninuki_renju": {},
    "pente": {},
    "connect4": {
        "backgroud-texture": "connect4",
        "stone-texture": "connect4",
        "squares": True,
    },
    "tictactoe": {
        "backgroud-texture": "tictactoe",
        "stone-texture": "tictactoe",
        "squares": True,
    },
    None: {},
}


def background_texture_path(theme: str) -> str:
    """
    Return the path of the background texture.
    """
    return f"assets/textures/{theme}_background.png"


def stone_texture_path(theme: str, color: str) -> str:
    """
    Return the path of the stone texture.
    """
    return f"assets/textures/{theme}_{color}_stone.png"


def group_arguments(parser: ArgumentParser, args: Namespace) -> Namespace:
    """
    Create new Namespace to store the values of the arguments by group.
    """
    args_by_group = Namespace()
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title not in ["positional arguments", "options"]:
            setattr(args_by_group, group.title, Namespace(**group_dict))
    return args_by_group


def organize_arguments(args: Namespace) -> Namespace:
    """
    Reorganize the values of the arguments to be more convenient to use.
    """

    def find_key(d):
        return next((k for k, v in d.items() if v), None)

    args.mode = find_key(vars(args.mode))
    engine_dict = vars(args.engine)
    depth = engine_dict.pop("depth", None)
    match find_key(engine_dict):
        case None:
            setattr(args, "engine", Namespace(algo="negamax", depth=depth))
        case arg_name:
            setattr(args, "engine", Namespace(algo=arg_name, depth=depth))

    # First elem is the padding since the playerID starts at 1
    args.player = (None, args.player.player1, args.player.player2)

    rules = vars(args.rules)
    if args.mode:
        rules.update(GAME_MODE_DEFAULT_RULES[args.mode])
    match rules["overline"]:
        case None:
            rules.update(overline=(None, False, False))
        case "p1":
            rules.update(overline=(None, True, False))
        case "p2":
            rules.update(overline=(None, False, True))
        case "both":
            rules.update(overline=(None, True, True))
    match rules["free_double"]:
        case None:
            rules.update(free_double=(None, False, False))
        case "p1":
            rules.update(free_double=(None, True, False))
        case "p2":
            rules.update(free_double=(None, False, True))
        case "both":
            rules.update(free_double=(None, True, True))
    args.rules = Namespace(**rules)

    display = vars(args.display)
    display.update(GAME_MODE_DEFAULT_THEME[args.mode])
    if args.display.background_color:
        display["background"] = tuple(args.display.background_color)
    else:
        display["background"] = background_texture_path(args.display.background_texture)
    display.pop("background_color", None)
    display.pop("background_texture", None)
    black_stone_path = stone_texture_path(display["stone_texture"], "black")
    white_stone_path = stone_texture_path(display["stone_texture"], "white")
    display["stone_texture"] = (None, black_stone_path, white_stone_path)
    display["line_color"] = tuple(args.display.line_color)
    args.display = Namespace(**display)
    return args


def parse_args(argv: list[str]) -> Namespace:
    """
    Create and parse the command line arguments by group
    """
    parser = ArgumentParser(prog="gomoku")

    engine_group = parser.add_argument_group(
        title="engine", description="Available algorithms and their depth"
    )
    engine_group.add_argument(
        "-d",
        "--depth",
        help="Maximal depth of the algorithm (default: 5)",
        type=int,
        default=5,
        metavar="N",
        choices=list(range(1, 16)),
    )
    engine_exclusive_group = engine_group.add_mutually_exclusive_group()
    engine_exclusive_group.add_argument(
        "--negamax", action="store_true", help="(Default) Simple negamax algorithm"
    )
    engine_exclusive_group.add_argument(
        "--ab-pruning",
        action="store_true",
        help="Variant of negamax algorithm called alpha-beta pruning",
    )
    engine_exclusive_group.add_argument(
        "--PVS",
        action="store_true",
        help="Variant of negamax algorithm called Principal Variation Search",
    )
    engine_exclusive_group.add_argument(
        "--MTD-f",
        action="store_true",
        help="Variant of ab pruning algorithm called MTD(f) or MTD(n,f)"
        ' for Memory-enhanced Test Driver with node "n" and value "f"',
    )
    engine_exclusive_group.add_argument(
        "--BNS",
        action="store_true",
        help="Variant of alpha-beta pruning algorithm called Best Node Search "
        "and originally known as fuzzified game tree search",
    )

    display_group = parser.add_argument_group(
        title="display", description="Display options for the user interface"
    )
    display_exclusive_group = display_group.add_mutually_exclusive_group()
    display_exclusive_group.add_argument(
        "--background-texture",
        help="The texture theme to use (default: classic)",
        choices={"classic", "connect4", "tictactoe"},
        default="classic",
    )
    display_exclusive_group.add_argument(
        "--background-color",
        help="RGB colors for the background. Overrides the background texture",
        nargs=3,
        type=int,
        metavar="N",
        choices=range(256),
    )
    display_group.add_argument(
        "--line-color",
        help="RGB colors for the lines",
        nargs=3,
        type=int,
        metavar="N",
        choices=range(256),
        default=[255, 255, 255],
    )
    display_group.add_argument(
        "--stone-texture",
        help="The texture theme to use for the stones",
        choices={"classic", "connect4", "tictactoe"},
        default="classic",
    )
    display_group.add_argument(
        "-u",
        "--user-interface",
        help="The user interface to use (default: gui)",
        choices={"gui", "cli"},
        default="gui",
    )
    display_group.add_argument(
        "-s",
        "--suggestion",
        action="store_true",
        help="Move suggestion for the human player",
    )
    display_group.add_argument(
        "--squares",
        help="Stones placed inside squares of the grid instead of at intersections",
        action="store_true",
    )

    player_group = parser.add_argument_group(
        title="player", description="Type selection for 1st and 2nd player"
    )
    player_group.add_argument(
        "-p1",
        "--player1",
        help="Player 1 (default: human)",
        choices={"human", "engine"},
        default="human",
    )
    player_group.add_argument(
        "-p2",
        "--player2",
        help="Player 2 (default: engine)",
        choices={"human", "engine"},
        default="engine",
    )

    rules_group = parser.add_argument_group(
        title="rules", description="List of available rules to configure the game"
    )
    rules_group.add_argument(
        "-b",
        "--board",
        nargs=2,
        choices=range(3, 26),
        type=int,
        default=[19, 19],
        metavar=("rows", "columns"),
        help="Size of the board (default: 19x19) (range: 3-25)",
    )
    rules_group.add_argument(
        "-w",
        "--winning-sequence",
        type=int,
        default=5,
        metavar="N",
        help="Number of consecutive pieces to win",
    )
    rules_group.add_argument(
        "-o",
        "--overline",
        help="Prohibits an overline of the winning sequence for a player",
        choices={"p1", "p2", "both"},
    )
    rules_group.add_argument(
        "-f",
        "--free-double",
        help="Prohibits an unstopable double sequence for a player",
        choices={"p1", "p2", "both"},
    )
    rules_group.add_argument(
        "-c",
        "--capture",
        nargs="?",
        type=int,
        metavar="N",
        choices=range(1, 101),
        help="Number of captures to win (range: 1-100)",
    )
    rules_group.add_argument(
        "-g",
        "--gravity",
        action="store_true",
        help="Enable gravity for the stones that will drop at the bottom row",
    )

    mode_group = parser.add_argument_group(
        title="mode",
        description="List of available game modes "
        "and their rules that override the rest of the rules",
    )
    mg = mode_group.add_mutually_exclusive_group()
    mg.add_argument(
        "--42",
        action="store_true",
        help="42 school variant (19x19) with (-f both) and (-c 5) applied",
    )
    mg.add_argument(
        "--freestyle", action="store_true", help="Variant (15x15) without restrictions"
    )
    mg.add_argument(
        "--renju", action="store_true", help="Variant (15x15) with (-f p1) and (-o p1)"
    )
    mg.add_argument(
        "--omok",
        action="store_true",
        help="Korean variant (19x19) with (-f both) applied",
    )
    mg.add_argument(
        "--ninuki-renju",
        action="store_true",
        help="Variant (15x15) with the same rules as renju and (-c 5)",
    )
    mg.add_argument(
        "--pente", action="store_true", help="Variant (19x19) with (-c 5) applied"
    )
    mg.add_argument(
        "--connect4",
        action="store_true",
        help="Connect4 game (6x7) (-g) and (-w 4) applied",
    )
    mg.add_argument(
        "--tictactoe",
        action="store_true",
        help="Tic-Tac-Toe game (3x3) with (-w 3) applied",
    )

    args = parser.parse_args(argv)
    args = group_arguments(parser, args)
    args = organize_arguments(args)

    return args
