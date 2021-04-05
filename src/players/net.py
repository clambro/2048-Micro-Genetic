from game.action import DIRECTIONS
import numpy as np
from players.base import Player


NUM_HIDDEN_LAYERS = 2
assert NUM_HIDDEN_LAYERS > 0
HIDDEN_LAYER_SIZE = 256
CHROMOSOME_SIZE = (17 * HIDDEN_LAYER_SIZE +  # Input layer weights
                   (NUM_HIDDEN_LAYERS - 1) * (HIDDEN_LAYER_SIZE + 1) * HIDDEN_LAYER_SIZE +  # Hidden layer weights
                   (HIDDEN_LAYER_SIZE + 1) * 4)  # Output layer weights


class NetworkPlayer(Player):
    """A simple feed-forward neural network to play 2048.

    Layers:
        1. Input layer of 16 nodes corresponding to the 16 tiles on the board.
        2. Fully-connected hidden layer of 16 nodes.
        3. Output layer of 4 nodes corresponding to left, up, right, down keys.

    Attributes
    ----------
    generation : int
        Which generation the network belongs to.
    chromosome : ndarray
        A 340x1 numpy array containing the weights for the matrices. (16 inputs; 1 hidden layer of size 16, plus bias.
        17x16 matrix then 17x4 = 340 elements)
    """

    def __init__(self, gen=0, mom=None, dad=None, chromosome=None):
        """Builds the network from a chromosome if given, or two parents, falling back to random generation if neither.

        Parameters
        ----------
        gen : int
            The current generation.
        mom : Optional[Net]
            A net from which the chromosome will be sampled.
        dad : Optional[Net]
            The other net from which the chromosome will be sampled.
        chromosome : Optional[ndarray]
            A 340x1 numpy array containing the network weights.
        """
        super().__init__()
        self.generation = gen
        if chromosome:
            self.chromosome = chromosome
        elif not mom or not dad:
            self.chromosome = 2 * np.random.randint(0, 2, CHROMOSOME_SIZE) - 1
        elif mom.get_avg_score() > dad.get_avg_score():
            self.chromosome = np.array([
                    mom.chromosome[i] if np.random.random() > 0.4
                    else dad.chromosome[i]
                    for i in range(len(mom.chromosome))
                    ])
            self._mutate()
        else:
            self.chromosome = np.array([
                    dad.chromosome[i] if np.random.random() > 0.4
                    else mom.chromosome[i]
                    for i in range(len(mom.chromosome))
                    ])
            self._mutate()

    def _mutate(self):
        """Add random mutations to 2% of net's chromosome."""
        mutation = np.array([-1 if np.random.random() < 0.01 else 1 for _ in range(len(self.chromosome))])
        self.chromosome *= mutation

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
        w_xh = self.chromosome[:17 * HIDDEN_LAYER_SIZE].reshape((17, HIDDEN_LAYER_SIZE))
        w_hh = self.chromosome[17 * HIDDEN_LAYER_SIZE:-(HIDDEN_LAYER_SIZE + 1) * 4]
        w_hh = w_hh.reshape((NUM_HIDDEN_LAYERS - 1, HIDDEN_LAYER_SIZE + 1, HIDDEN_LAYER_SIZE))
        w_hy = self.chromosome[-(HIDDEN_LAYER_SIZE + 1) * 4:].reshape((HIDDEN_LAYER_SIZE + 1, 4))

        x = board.reshape(16) / np.max(board)  # Only relative tile magnitude matters.
        x = np.append(1, x)  # Add bias
        h = np.sign(x @ w_xh)
        for w in w_hh:
            h = np.append(1, h)
            h = np.sign(h @ w)
        h = np.append(1, h)
        y = h @ w_hy  # No non-linearity needed. We only care about order.
        return np.asarray(DIRECTIONS)[y.argsort()[::-1]]
