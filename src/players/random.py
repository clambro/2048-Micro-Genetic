import numpy as np
from players.base import Player


class RandomPlayer(Player):
    """Play 2048 by moving randomly."""

    def __init__(self):
        super().__init__()

    def _choose_action(self, game):
        """Return a random legal move.

        Parameters
        ----------
        game : Game
            The current game state.

        Returns
        -------
        Action
            The action to take.
        """
        return np.random.choice(game.get_legal_moves())
