from typing import Iterator

Coord = tuple[int, int]


def add(a: Coord, b: Coord) -> Coord:
    """
    Add two tuples of coordinates
    """
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Coord, b: Coord) -> Coord:
    """
    Subtract two tuples of coordinates
    """
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Coord, x: int) -> Coord:
    """
    Multiply a tuple of coordinates by a scalar
    """
    return (a[0] * x, a[1] * x)


def neg(a: Coord) -> Coord:
    """
    Negate a tuple of coordinates
    """
    return (-a[0], -a[1])


def distance(a: Coord, b: Coord) -> int:
    """
    Compute the distance between two coordinates
    """
    d = sub(a, b)
    return max(abs(d[0]), abs(d[1]))


def range_coord(start: Coord, dir: Coord, length: int) -> Iterator[Coord]:
    """
    Generate a range of coordinates
    """
    for i in range(length):
        yield add(start, mul(dir, i))


def range_shape(start: Coord, dir: Coord, shape: tuple[int, ...]) -> Iterator[Coord]:
    """
    Takes a shape (length of subsequences separated by an empty cell) and gets
    the range of Coords for each len in shape while skipping cells in between.
    """
    current = start
    for length in shape:
        for coord in range_coord(current, dir, length):
            current = coord
            yield coord
        current = add(current, mul(dir, 2))


def in_bound(t: Coord, size: int) -> bool:
    """
    Returns True if the Coord is between a certain bound.
    """
    return t and 0 <= t[0] < size and 0 <= t[1] < size
