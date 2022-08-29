import numpy as np
from random import randint

BOARD_SIZE = 10
board = np.arange(BOARD_SIZE * BOARD_SIZE).reshape(BOARD_SIZE, BOARD_SIZE)

print(board)
y, x = (0, 0)
iy, ix = (BOARD_SIZE - 1 - y, BOARD_SIZE - 1 - x)
print(board[y, x])

# Left
print(f"← (left): {board[y, max(x - 4, 0):x + 1][::-1]}")
# Right
print(f"→ (right): {board[y, x:x + 5]}")
# Up
print(f"↑ (up): {board[max(y - 4, 0):y + 1, x][::-1]}")
# Down
print(f"↓ (down): {board[y:y + 5, x]}")

# Diagonal down-right
print(f"↘ (down-right): {np.diagonal(board[y:, x:])[:5]}")
# Diagonal down-left
print(f"↙ (down-left): {np.fliplr(board)[y:, ix:].diagonal()[:5]}")
# Diagonal up-right
print(f"↗ (up-right): {np.flipud(board)[iy:, x:].diagonal()[:5]}")
# Diagonal up-left
print(f"↖ (up-left): {np.fliplr(np.flipud(board))[iy:, ix:].diagonal()[:5]}")