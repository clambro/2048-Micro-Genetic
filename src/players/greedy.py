from copy import deepcopy
from game.action import Action
import numpy as np
from players.base import Player


class GreedyPlayer(Player):
    """Play 2048 by choosing the move that maximizes score at every step."""

    def __init__(self):
        super().__init__()

    def _choose_action(self, game):
        """Choose the legal move that leads to the highest score.

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
        if len(legal_moves) == 1:
            return legal_moves[0]
        scores = []
        for move in legal_moves:
            g = deepcopy(game)
            g.move(move)
            scores.append(g.score)
        return legal_moves[np.argmax(scores)]
