from game.action import DIRECTIONS
from genetics.genome import Genome
import numpy as np
from players.base import Player


class BinaryNetworkPlayer(Player):
    """A simple feed-forward neural network to play 2048.

    Layers:
        1. Input layer of 16 nodes corresponding to the 16 tiles on the board.
        2. Fully-connected hidden layer of 16 nodes.
        3. Output layer of 4 nodes corresponding to left, up, right, down keys.

    Attributes
    ----------
    generation : int
        Which generation the network belongs to.
    genome : ndarray
        A 340x1 numpy array containing the weights for the matrices. (16 inputs; 1 hidden layer of size 16, plus bias.
        17x16 matrix then 17x4 = 340 elements)
    """

    def __init__(self, gen=0, mom=None, dad=None, genome=None):
        """Builds the network from a chromosome if given, or two parents, falling back to random generation if neither.

        Parameters
        ----------
        gen : int
            The current generation.
        mom : Optional[Net]
            A net from which the chromosome will be sampled.
        dad : Optional[Net]
            The other net from which the chromosome will be sampled.
        genome : Optional[ndarray]
            A 340x1 numpy array containing the network weights.
        """
        super().__init__()
        self.generation = gen
        if genome is not None:
            self.genome = genome
        elif None not in [mom, dad]:
            self.genome = Genome(mom.genome, dad.genome)
        else:
            self.genome = Genome()

    def _choose_action(self, game):
        """Evaluate the position using the network and choose the best legal move it determines.

        Parameters
        ----------
        game : Game
            The current game state.

        Returns
        -------
        best_move : Action
            The action to take.
        """
        legal_moves = game.get_legal_moves()
        sorted_moves = self._get_network_move_order(game.board)
        for move in sorted_moves:
            if move in legal_moves:
                return move

    def _get_network_move_order(self, board):
        """Input board into the network and evaluate it to get a move direction.

        Parameters
        ----------
        board : ndarray
            The board state to calculate the move for.

        Returns
        -------
        ndarray
            The four direction actions sorted in the order of the network's evaluation.
        """
        x = 3 * (board.reshape(16) / 7 - 1)  # Max tile log-value in 2048 is 14. Normalize to [-3, 3].
        h = np.sign(x @ self.genome.input_weights)
        for w in self.genome.hidden_weights:
            h = np.sign(h @ w)
        y = h @ self.genome.output_weights  # No non-linearity needed. We only care about order.
        return np.asarray(DIRECTIONS)[y.argsort()[::-1]]
