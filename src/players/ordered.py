from game.action import Action
from players.base import Player


class OrderedPlayer(Player):
    """Play 2048 by using a simple heuristic to push everything to the bottom-right.

    Attributes
    ----------
    previous_action : Action
        The action taken in the previous game position.
    """

    def __init__(self):
        """Initialize the player and set the previous action to None."""
        super().__init__()
        self.previous_action = None

    def _choose_action(self, game):
        """Alternate moving down and to the right, choosing the first legal move if neither is an option.

        Parameters
        ----------
        game : Game
            The current game state.

        Returns
        -------
        Action
            The action to take.
        """
        legal_moves = game.get_legal_moves()
        if self.previous_action != Action.DOWN and Action.DOWN in legal_moves:
            self.previous_action = Action.DOWN
        elif Action.RIGHT in legal_moves:
            self.previous_action = Action.RIGHT
        else:
            self.previous_action = legal_moves[0]
        return self.previous_action
