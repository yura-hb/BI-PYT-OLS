# Homework 01 - Game of life
# 
# Your task is to implement part of the cell automata called
# Game of life. The automata is a 2D simulation where each cell
# on the grid is either dead or alive.
# 
# State of each cell is updated in every iteration based state of neighbouring cells.
# Cell neighbours are cells that are horizontally, vertically, or diagonally adjacent.
#
# Rules for update are as follows:
# 
# 1. Any live cell with fewer than two live neighbours dies, as if by underpopulation.
# 2. Any live cell with two or three live neighbours lives on to the next generation.
# 3. Any live cell with more than three live neighbours dies, as if by overpopulation.
# 4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
#
# 
# Our implementation will use coordinate system will use grid coordinates starting from (0, 0) - upper left corner.
# The first coordinate is row and second is column.
# 
# Do not use wrap around (toroid) when reaching edge of the board.
# 
# For more details about Game of Life, see Wikipedia - https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

def neighboursCount(item, alive):
    dist = [ (i, j) for i in range(-1, 2, 1) for j in range(-1, 2, 1) ]
    dist.remove((0, 0))
    return len([ (item[0] + d[0], item[1] + d[1]) for d in dist if (item[0] + d[0], item[1] + d[1]) in alive])


def update(alive, size, iter_n):
    height, width = size
    for _ in range(iter_n):
        alive = set(
            [cell for cell in alive if neighboursCount(cell, alive) in [2, 3]] +
            [ (j, i) for i in range(width) for j in range(height) if (j, i) not in alive and neighboursCount((j, i), alive) == 3]
        )
    return alive

def draw(alive, size):
    """
    alive - set of cell coordinates marked as alive, can be empty
    size - size of simulation grid as  tuple - (

    output - string showing the board state with alive cells marked with X
    """
    height, width = size
    dic = { i[0] * height + i[1] : True for i in alive }
    border = '+' + ''.join([ '-' for i in range(width)]) + '+'
    board = '\n'.join(['|' + ''.join([ 'X' if (j, i) in alive else ' ' for i in range(width) ]) + '|' for j in range(height)])
    return border + '\n' + board + '\n' + border

