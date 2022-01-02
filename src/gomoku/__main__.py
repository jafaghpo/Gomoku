import sys
from argparse import Namespace, ArgumentParser

def parse_args() -> Namespace:
    parser = ArgumentParser(prog='gomoku')
    engine_group = parser.add_argument_group('engine')
    engine_group.add_argument('-a', '--algo', help='Algorithm to use',
        choices={'negamax', 'alphabeta', 'PVS', 'MTD', 'BNS', 'MCTS'}, default='BNS')
    engine_group.add_argument('-d', '--depth', help='maximal depth of the algorithm',
        type=int, default=5, metavar='N', choices=list(range(1, 11)))

    display_group = parser.add_argument_group('display')
    display_group.add_argument('-g', '--gui', action='store_true',
        help='Use the graphical user interface instead of the console')
    display_group.add_argument('-s', '--suggestion', action='store_true',
        help='Move suggestion for the human player')

    player_group = parser.add_argument_group('player')
    player_group.add_argument('-p1', '--player1', help='Player 1',
        choices={'human', 'engine'}, default='human')
    player_group.add_argument('-p2', '--player2', help='Player 2',
        choices={'human', 'engine'}, default='engine')

    rules_group = parser.add_argument_group('rules')    
    rules_group.add_argument('-b', '--board', nargs=2, type=int, default=[19, 19],
        metavar=('columns', 'rows'), help='Size of the board')
    rules_group.add_argument('-w', '--winning-sequence', type=int, default=5,
        metavar='N', help='Number of consecutive pieces to win')
    rules_group.add_argument('-o', '--overline',
        help='Enable to have an overline of the winning sequence for a specific player',
        choices={'player1', 'player2', 'both'}, default='both')
    rules_group.add_argument('-f', '--free-sequence',
        help='Allow to have an unstopable double sequence for a specific player',
        choices={'player1', 'player2', 'both'}, default='both')
    rules_group.add_argument('-c', '--capture', type=int, default=5, metavar='N',
        help='Number of captures to win')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
