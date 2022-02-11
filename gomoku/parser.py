from argparse import ArgumentParser, Namespace

def group_arguments(parser: ArgumentParser, args: Namespace) -> Namespace:
    """
    Create a new Namespace to store the values of the arguments by group.
    Arguments:
        - parser: the ArgumentParser object used to get the names of the groups
        - args: a Namespace object containing the values of command line arguments
    Returns:
        - a new Namespace object with arguments grouped by group of arguments
    """
    args_by_group = Namespace()
    for group in parser._action_groups:
        group_dict =  { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
        if group.title not in ['positional arguments', 'options']:
            setattr(args_by_group, group.title, Namespace(**group_dict))
    return args_by_group

def set_default_values(args: Namespace) -> Namespace:
    """
    Assign the argument selected by the user the 'mode' and 'engine' group
    or a default value if not specified.
    Arguments:
        - args: a Namespace object containing the values of command line arguments by group
    Returns:
        - a new Namespace object with a str containing the name of the mode for the 'mode' group
         and a Namespace containing the 'engine' group
    """
    
    # Transform the Namespace object into a dictionnary and
    # return the first value equal to True or None if no such value is found
    find_key = lambda d: next((k for k, v in vars(d).items() if v == True), None)

    args.mode = Namespace(name=find_key(args.mode))
    match find_key(args.engine):
        case None: setattr(args, 'engine', Namespace(algo_name='negamax', depth=args.engine.depth))
        case arg_name: setattr(args, 'engine', Namespace(algo_name=arg_name, depth=args.engine.depth))
    return args

def parse_args(argv: list[str]) -> Namespace:
    """
    Create and parse the command line arguments by group
    Arguments:
        - argv: a list of str containing the command line arguments 
    Returns:
        - a Namespace object regrouping the values of the command line arguments by group
    """
    parser = ArgumentParser(prog='gomoku')

    engine_group = parser.add_argument_group(title='engine', description='Available algorithms and their depth to configure the engine')
    engine_group.add_argument('-d', '--depth', help='Maximal depth of the algorithm (default: 5)',
        type=int, default=5, metavar='N', choices=list(range(1, 16)))
    eg = engine_group.add_mutually_exclusive_group()
    eg.add_argument('--negamax', action='store_true', help='(Default) Simple negamax algorithm')
    eg.add_argument('--ab-pruning', action='store_true',
        help='Variant of negamax algorithm called alpha-beta pruning')
    eg.add_argument('--PVS', action='store_true', help='Variant of negamax algorithm '
        'called Principal Variation Search')
    eg.add_argument('--MTD-f', action='store_true', help='Variant of alpha-beta pruning algorithm '
        'called MTD(f) or MTD(n,f) for Memory-enhanced Test Driver with node "n" and value "f"')
    eg.add_argument('--BNS', action='store_true', help='Variant of alpha-beta pruning algorithm '
        'called Best Node Search and originally known as fuzzified game tree search')

    display_group = parser.add_argument_group(title='display', description='Display options for the user interface')
    display_group.add_argument('-u', '--user-interface', help='The user interface to use (default: gui)',
        choices={'gui', 'cli'}, default='gui')
    display_group.add_argument('-s', '--suggestion', action='store_true',
        help='Move suggestion for the human player')
    display_group.add_argument('-t', '--theme', help='The texture theme to use (default: classic)',
        choices={'classic', 'wood', 'dark', 'space', 'connect4', 'tictactoe'}, default='classic')
    display_group.add_argument('-p', '--placement', help='The placement type of the board (default: intersection)',
        choices={'intersection', 'square'}, default='intersection')

    player_group = parser.add_argument_group(title='player', description='Type selection for the first and second player')
    player_group.add_argument('-p1', '--player1', help='Player 1 (default: human)',
        choices={'human', 'engine'}, default='human')
    player_group.add_argument('-p2', '--player2', help='Player 2 (default: engine)',
        choices={'human', 'engine'}, default='engine')

    rules_group = parser.add_argument_group(title='rules', description='List of available rules to configure the game')
    rules_group.add_argument('-b', '--board', nargs=2, choices=range(3, 26), type=int, default=[19, 19],
        metavar=('columns', 'rows'), help='Size of the board (default: 19x19) (range: 3-25)')
    rules_group.add_argument('-w', '--winning-sequence', type=int, default=5,
        metavar='N', help='Number of consecutive pieces to win')
    rules_group.add_argument('-o', '--overline',
        help='Enable to have an overline of the winning sequence for a specific player',
        choices={'p1', 'p2', 'both'})
    rules_group.add_argument('-f', '--free-double',
        help='Enable to have an unstopable double sequence for a specific player',
        choices={'p1', 'p2', 'both'})
    rules_group.add_argument('-c', '--capture', nargs='?', type=int, metavar='N',
        choices=range(1, 101), help='Number of captures to win (range: 1-100)')
    rules_group.add_argument('-g', '--gravity', action='store_true',
        help='Enable gravity for the stones that will drop at the bottom row')
    
    mode_group = parser.add_argument_group(title='mode', description='List of available game modes '
        'and their rules that override the rest of the rules')
    mg = mode_group.add_mutually_exclusive_group()
    mg.add_argument('--42', action='store_true',
        help='(Default) 42 school variant of gomoku (19x19) with (-o both) and (-c 5) applied')
    mg.add_argument('--freestyle', action='store_true',
        help='Variant of gomoku (15x15) with (-f both) and (-o both) applied')
    mg.add_argument('--renju', action='store_true',
        help='Variant of gomoku (15x15) with (-o p2)')
    mg.add_argument('--omok', action='store_true',
        help='Korean variant of gomoku (19x19) with (-o both) applied')
    mg.add_argument('--ninuki-renju', action='store_true',
        help='Variant of gomoku (15x15) with the same rules as renju and (-c 5)')
    mg.add_argument('--pente', action='store_true',
        help='Variant of gomoku (19x19) with (-f both), (-o both) and (-c 5) applied')
    mg.add_argument('--connect4', action='store_true',
        help='Connect4 game (7x6) that use the gravity rule (-g) '
        'with (-f both) and (-o both) applied')
    mg.add_argument('--tictactoe', action='store_true',
        help='Tic-Tac-Toe game (3x3) with (-f both) applied')
    
    args = parser.parse_args(argv)
    args = group_arguments(parser, args)
    args = set_default_values(args)
 
    return args