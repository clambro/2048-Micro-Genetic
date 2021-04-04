from enum import Enum, auto


class Action(Enum):
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    QUIT = auto()


DIRECTIONS = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
