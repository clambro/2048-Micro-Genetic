import numpy as np


HIDDEN_LAYER_SIZE = 256
NUM_HIDDEN_LAYERS = 2
assert NUM_HIDDEN_LAYERS > 0

INPUT_WEIGHT_SHAPE = (16, HIDDEN_LAYER_SIZE)
HIDDEN_WEIGHTS_SHAPE = ((NUM_HIDDEN_LAYERS - 1), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
OUTPUT_WEIGHT_SHAPE = (HIDDEN_LAYER_SIZE, 4)

CHROMOSOME_SIZE = sum(np.prod(w) for w in [INPUT_WEIGHT_SHAPE, HIDDEN_WEIGHTS_SHAPE, OUTPUT_WEIGHT_SHAPE])


class Genome:
    """"""

    def __init__(self, mom=None, dad=None):
        if None not in [mom, dad]:
            self.chromosome = np.array([
                    mom.chromosome[i] if np.random.random() > 0.5
                    else dad.chromosome[i]
                    for i in range(len(mom.chromosome))
                    ])
            self._mutate()
        else:
            self.chromosome = 2 * np.random.randint(0, 2, CHROMOSOME_SIZE) - 1

    def _mutate(self):
        """Add random mutations to 2% of net's chromosome."""
        mutation = np.array([-1 if np.random.random() < 0.01 else 1 for _ in range(len(self.chromosome))])
        self.chromosome *= mutation

    def get_weight_matrices(self):
        """"""
        w_xh = self.chromosome[:np.prod(INPUT_WEIGHT_SHAPE)].reshape(INPUT_WEIGHT_SHAPE)
        w_hh = self.chromosome[np.prod(INPUT_WEIGHT_SHAPE):-np.prod(OUTPUT_WEIGHT_SHAPE)].reshape(HIDDEN_WEIGHTS_SHAPE)
        w_hy = self.chromosome[-np.prod(OUTPUT_WEIGHT_SHAPE):].reshape(OUTPUT_WEIGHT_SHAPE)
        return w_xh, w_hh, w_hy
