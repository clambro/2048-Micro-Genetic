from game import DIRECTIONS
import numpy as np


HIDDEN_LAYER_SIZE = 1024
NUM_HIDDEN_LAYERS = 1
assert NUM_HIDDEN_LAYERS > 0

INPUT_WEIGHT_SHAPE = (16, HIDDEN_LAYER_SIZE)
HIDDEN_WEIGHTS_SHAPE = ((NUM_HIDDEN_LAYERS - 1), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
OUTPUT_WEIGHT_SHAPE = (HIDDEN_LAYER_SIZE, 4)


class Genome:
    """The weights for a binary neural network for NetworkPlayer with rules for reproduction.

    Attributes
    ----------
    input_weights : ndarray
        The weights for the first layer with shape INPUT_WEIGHT_SHAPE.
    hidden_weights : ndarray
        The weights for all the hidden layers with shape HIDDEN_WEIGHTS_SHAPE.
    output_weights : ndarray
        The weights for the final layer with shape OUTPUT_WEIGHT_SHAPE.
    """

    def __init__(self, mom=None, dad=None):
        """Initializes the genome either through reproduction from parents or random generation.

        Parameters
        ----------
        mom : Optional[Genome]
            The first of the two parent genomes.
        dad : Optional[Genome]
            The second of the two parent genomes.
        """
        if None not in [mom, dad]:
            self.input_weights, self.hidden_weights, self.output_weights = self._spawn_child_chromosome(mom, dad)
        else:
            def generate_binary_weights(shape):
                """Generate binary {-1, 1} weights of a given shape."""
                return 2 * np.random.randint(0, 2, shape) - 1
            self.input_weights = generate_binary_weights(INPUT_WEIGHT_SHAPE)
            self.hidden_weights = generate_binary_weights(HIDDEN_WEIGHTS_SHAPE)
            self.output_weights = generate_binary_weights(OUTPUT_WEIGHT_SHAPE)

    @staticmethod
    def _spawn_child_chromosome(mom, dad):
        """Spawn mutated weight arrays from two parent Genomes.

        Weights are passed down to children one matrix row at a time to preserve some similarity between parents and
        offspring. Mutations happen on a per-weight basis, however.

        Parameters
        ----------
        mom : Genome
            The first of the two parent genomes.
        dad : Genome
            The second of the two parent genomes.

        Returns
        -------
        input_weights : ndarray
            The weights for the first layer with shape INPUT_WEIGHT_SHAPE.
        hidden_weights : ndarray
            The weights for all the hidden layers with shape HIDDEN_WEIGHTS_SHAPE.
        output_weights : ndarray
            The weights for the final layer with shape OUTPUT_WEIGHT_SHAPE.
        """
        input_weights = np.array([m if np.random.random() > 0.5 else d
                                  for m, d in zip(mom.input_weights, dad.input_weights)])

        hidden_weights = []
        for m_hid, d_hid in zip(mom.hidden_weights, dad.hidden_weights):
            hidden_weights.append(np.array([m if np.random.random() > 0.5 else d for m, d in zip(m_hid, d_hid)]))
        hidden_weights = np.asarray(hidden_weights)

        output_weights = np.array([m if np.random.random() > 0.5 else d
                                   for m, d in zip(mom.output_weights, dad.output_weights)])

        def mutate(array):
            """Randomly flip or zero ~1% of the bits."""
            mutation = np.array([np.random.choice([-1, 0, 1])
                                 if np.random.random() < 0.01 else i for i in array.reshape(-1)])
            return mutation.reshape(array.shape)

        return mutate(input_weights), mutate(hidden_weights), mutate(output_weights)

    def calculate_move_order(self, board):
        """Input board into the network and evaluate it to get the priority for each move direction.

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
        h = np.sign(x @ self.input_weights)
        for w in self.hidden_weights:
            h = np.sign(h @ w)
        y = h @ self.output_weights  # No non-linearity needed. We only care about order.
        return np.asarray(DIRECTIONS)[y.argsort()[::-1]]

    def calculate_similarity(self, genome):
        """Calculate the similarity (percent of equal weights) between this genome and another.

        Parameters
        ----------
        genome : Genome
            The genome to which this one will be compared.

        Returns
        -------
        float
            The percent similarity from 0 to 1.
        """
        w1 = np.hstack([w.reshape(-1) for w in (self.input_weights, self.hidden_weights, self.output_weights)])
        w2 = np.hstack([w.reshape(-1) for w in (genome.input_weights, genome.hidden_weights, genome.output_weights)])
        return np.mean(w1 == w2)
