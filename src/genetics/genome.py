import numpy as np


HIDDEN_LAYER_SIZE = 256
NUM_HIDDEN_LAYERS = 2
assert NUM_HIDDEN_LAYERS > 0

INPUT_WEIGHT_SHAPE = (16, HIDDEN_LAYER_SIZE)
HIDDEN_WEIGHTS_SHAPE = ((NUM_HIDDEN_LAYERS - 1), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
OUTPUT_WEIGHT_SHAPE = (HIDDEN_LAYER_SIZE, 4)


class Genome:
    """"""

    def __init__(self, mom=None, dad=None):
        if None not in [mom, dad]:
            self.input_weights, self.hidden_weights, self.output_weights = self._spawn_child_chromosome(mom, dad)
        else:
            self.input_weights = self._generate_random_weights(INPUT_WEIGHT_SHAPE)
            self.hidden_weights = self._generate_random_weights(HIDDEN_WEIGHTS_SHAPE)
            self.output_weights = self._generate_random_weights(OUTPUT_WEIGHT_SHAPE)

    @staticmethod
    def _generate_random_weights(shape):
        """"""
        return 2 * np.random.randint(0, 2, shape) - 1

    @staticmethod
    def _spawn_child_chromosome(mom, dad):
        """"""
        input_weights = np.array([m if np.random.random() > 0.5 else d
                                  for m, d in zip(mom.input_weights, dad.input_weights)])

        hidden_weights = []
        for m_hid, d_hid in zip(mom.hidden_weights, dad.hidden_weights):
            hidden_weights.append(np.array([m if np.random.random() > 0.5 else d for m, d in zip(m_hid, d_hid)]))
        hidden_weights = np.asarray(hidden_weights)

        output_weights = np.array([m if np.random.random() > 0.5 else d
                                   for m, d in zip(mom.output_weights, dad.output_weights)])

        def mutate(array):
            mutation = np.array([-1 if np.random.random() < 0.01 else 1 for _ in range(array.size)])
            return array * mutation.reshape(array.shape)

        return mutate(input_weights), mutate(hidden_weights), mutate(output_weights)
