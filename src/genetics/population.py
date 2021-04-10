import matplotlib.pyplot as plt
import pickle
from players.net import NetworkPlayer
import numpy as np


NETS_PER_POP = 32
NUM_ELITE = 4


class Population:
    """A collection of networks.

    Attributes
    ----------
    generation : int
        The current generation.
    networks : List[NetworkPlayer]
        The networks in the population.
    elites : List[NetworkPlayer]
        Additional elite networks from a previous generation that may be treated differently.
    """

    def __init__(self, pop=None):
        """Builds the population by either reproducing from a previous one or randomly generating networks.

        Parameters
        ----------
        pop : Union[Population, str]
            The population from which to spawn this population, or a path leading to it. If None, the population will
            be generated randomly.
        """
        if pop is None:
            self.generation = 1
            self.elites = []
            self.networks = [NetworkPlayer() for _ in range(NETS_PER_POP)]
        else:
            if isinstance(pop, str):
                with open(pop, 'rb') as f:
                    pop = pickle.load(f)
            self.generation = pop.generation + 1
            prev_networks = pop.get_sorted_networks(include_elite=True)
            self.elites = prev_networks[:NUM_ELITE]
            self.networks = self._spawn_children(prev_networks)

    @staticmethod
    def _spawn_children(parents):
        """Generate a list of child networks from a list of parents.

        Parameters
        ----------
        parents : List[NetworkPlayer]
            Networks from which the population will be spawned.

        Returns
        -------
        List[NetworkPlayer]
            The networks for the current population.
        """
        moms = np.random.choice(parents, NETS_PER_POP - NUM_ELITE)
        dads = np.random.choice(parents, NETS_PER_POP - NUM_ELITE)
        return [NetworkPlayer(mom=m, dad=d) for m, d in zip(moms, dads)]

    def play_games(self, games, include_elites):
        """Get each network in the population to play a certain number of games.

        Parameters
        ----------
        games : int
            The number of games each network should play.
        """
        [n.play_multiple_games(games) for n in self.networks]
        if include_elites:
            [n.play_multiple_games(games) for n in self.elites]

    def get_sorted_networks(self, include_elites):
        """Sort the genepool in descending order by each network's average score."""
        networks = self.networks
        if include_elites:
            networks += self.elites
        return networks.sort(key=lambda n: n.get_avg_score(), reverse=True)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


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
    top_scores : List[float]
        List of floats. The top average score in each generation.
    best_net : NetworkPlayer
        The trained networks that performs best.
    """
    parents = None
    top_scores = []

    for gen in range(initial_gen+1, final_gen+1):
        pop = Population(gen, elite, parents)

        print('Playing games for generation', gen, 'of', final_gen)
        pop.play_games()
        pop.sort_by_score()
        if not pop.generation % 20 and pop.generation != 0:
            name = 'Generation' + str(pop.generation)
            np.save(name, pop.genepool[:2])

        top_scores.append(pop.genepool[0].get_avg_score())

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
    plt.plot(np.log2(top_scores))

    filename = "BestNetGen" + str(final_gen)
    np.save(filename, pop.genepool[0])

    return top_scores, pop.genepool[0]
