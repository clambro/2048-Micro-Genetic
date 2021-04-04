from game.action import DIRECTIONS
import numpy as np
from players.base import Player


class Net(Player):
    """A simple feed-forward neural net to play 2048.

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
            self.chromosome = np.random.uniform(-0.4, 0.4, 340)
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
        mutation = np.array([np.random.randn()/10 if np.random.random() < 0.02
                             else 0 for _ in range(len(self.chromosome))])
        self.chromosome += mutation

    def _choose_action(self, game):
        """"""
        legal_moves = game.get_legal_moves()
        best_move = None
        highest_priority = -np.inf
        for direction, priority in zip(DIRECTIONS, self._predict_move(game.board)):
            if direction in legal_moves and priority > highest_priority:
                best_move = direction
                highest_priority = highest_priority
        return best_move

    def _predict_move(self, board):
        """Input board into net and feed-forward to get a move direction.

        Parameters
        ----------
        board : ndarray
            The board state to calculate the move for.

        Returns
        -------
        moves : ndarray
            A list of integers corresponding to movement directions, sorted according to the net's output.
        """
        x = board.reshape(16) / np.max(board)  # Only relative tile magnitude matters.
        x = np.append(1, x)  # Add bias
        w_xh = self.chromosome[:272].reshape((17, 16))
        w_hy = self.chromosome[272:].reshape((17, 4))
        h = x @ w_xh
        h = np.maximum(0.01 * h, h)  # Leaky ReLU
        h = np.append(1, h)  # Add bias
        return (h @ w_hy).tolist()  # No non-linearity needed. We only care about order.
