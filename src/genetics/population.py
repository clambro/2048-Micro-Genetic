import matplotlib.pyplot as plt
from players.net import NetworkPlayer
import numpy as np


class Population:
    """A collection of networks.

    Attributes
    ----------
    generation : int
        The current generation.
    genepool : List[Net]
        The networks in the population.
    """

    def __init__(self, gen, elite=None, parents=None):
        """Builds the population by copying elites and creating children from the genepool or randomly generating them.

        Parameters
        ----------
        gen :  int
            The generation of new networks.
        elite : Union[List[Net], str]
            Networks that are copied directly to the population OR a file path to such a list.
        parents : List[Net]
            Networks from which the population will be spawned.
        """
        self.generation = gen

        if isinstance(elite, str):
            self.genepool = np.load(elite).tolist()
        elif elite is not None:
            self.genepool = elite
        else:
            self.genepool = []

        if parents is not None:
            moms, dads = self._tournament(parents)
            for i in range(8):
                self.genepool.append(NetworkPlayer(gen, moms[i], dads[i]))
        else:
            current = len(self.genepool)
            for i in range(10 - current):
                self.genepool.append(NetworkPlayer(gen))

    @staticmethod
    def _tournament(parents):
        """Tournament selection to generate pairs of parents.

        The network with the highest score is selected to be a parent with probability p, then second highest p*(p-1),
        third highest p*(p-1)**2, etc.

        Parameters
        ----------
        parents : List[Net]
            Networks from which the population will be spawned.

        Returns
        -------
        moms : List[Net]
            A list of 8 networks.
        dads : List[Net]
            A list of 8 networks.
        """
        moms = []
        dads = []
        while len(moms) < 8:
            m1, m2, d1, d2 = np.random.choice(parents, 4)
            if m1.get_avg_score() > m2.get_avg_score():
                moms.append(m1)
            else:
                moms.append(m2)
            if d1.get_avg_score() > d2.get_avg_score():
                dads.append(d1)
            else:
                dads.append(d2)
        return moms, dads

    def play_games(self, games=50):
        """Get each network in the population to play a certain number of games.

        Parameters
        ----------
        games : int
            The number of games each network should play.
        """
        for n in self.genepool:
            n.play_multiple_games(games)

    def sort_by_score(self):
        """Sort the genepool in descending order by each network's average score."""
        self.genepool.sort(key=lambda n: n.get_avg_score(), reverse=True)


def train_population(final_gen, initial_gen=0, elite=None):
    """Run a micro-genetic algorithm to evolve a good neural network.

    Each population defaults to 10 networks and plays 50 games. The top 2 from each generation are copied to the next
    one, but all have the opportunity to reproduce. Every 10 generations, all but the top 2 are killed and 8 new
    networks are randomly generated to add to the 2 that survived.

    Parameters
    ----------
    final_gen : int
        The total number of generations played is final_gen - initial_gen.  Recommended to be 10*n for positive integer
        values of n.
    initial_gen : int
        The total number of generations played is final_gen - initial_gen
    elite : List[int]
        List of networks to copy directly into the new population.

    Returns
    -------
    top_tiles : List[float]
        List of floats. The top average score in each generation.
    best_net : NetworkPlayer
        The trained networks that performs best.
    """
    parents = None
    top_tiles = []

    for gen in range(initial_gen+1, final_gen+1):
        pop = Population(gen, elite, parents)

        print('Playing games for generation', gen, 'of', final_gen)
        pop.play_games()
        pop.sort_by_score()
        if not pop.generation % 20 and pop.generation != 0:
            name = 'Generation' + str(pop.generation)
            np.save(name, pop.genepool[:2])

        top_tiles.append(pop.genepool[0].get_avg_score())

        elite = pop.genepool[:2]

        if not gen % 10:
            parents = None
        else:
            parents = pop.genepool

        print('Best network\'s generation =', pop.genepool[0].generation)
        print('Best network\'s score =', np.rint(pop.genepool[0].get_avg_score()))
        print('Best network\'s highest tile =', np.rint(pop.genepool[0].get_avg_highest_tile()), '\n')

    plt.figure()
    plt.title('log_2(Highest Score)')
    plt.plot(np.log2(top_tiles))

    filename = "BestNetGen" + str(final_gen)
    np.save(filename, pop.genepool[0])

    return top_tiles, pop.genepool[0]
