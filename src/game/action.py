from enum import Enum, auto


class Action(Enum):
    """Enumerates the legal actions in 2048: the four directions and the option to quit."""
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    QUIT = auto()


DIRECTIONS = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
