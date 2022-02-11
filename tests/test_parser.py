from gomoku.parser import parse_args
import pytest

def test_parse_args_1():
    argv = ['--player2', 'human', '--PVS', '--depth', '7', '-s', '-b', '4', '3']
    args = parse_args(argv)
    assert args.player.player2 == 'human'
    assert args.engine.algo_name == 'PVS'
    assert args.engine.depth == 7
    assert args.display.suggestion == True
    assert args.rules.board == [4, 3]

def test_parse_args_2():
    argv = ['--player1', 'engine', '--negamax', '--depth', '16', '-b', '26', '4']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parse_args(argv)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2