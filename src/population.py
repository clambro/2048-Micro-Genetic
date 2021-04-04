import matplotlib.pyplot as plt
import net
import numpy as np
from os import remove
from os.path import isfile


class Population:
    """A collection of nets.

    Attributes
    ----------
    generation : int
        The current generation.
    genepool : List[Net]
        The nets in the population.
    """

    def __init__(self, gen, elite=None, parents=None):
        """Builds the population by copying elites and creating children from the genepool or randomly generating them.

        Parameters
        ----------
        gen :  int
            The generation of new nets.
        elite : Union[List[Net], str]
            Nets that are copied directly to the population OR a file name corresponding to a file containing that list.
        parents : List[Net]
            Nets from which the population will be spawned.
        """
        self.generation = gen

        if isinstance(elite, str):
            self.genepool = np.load(elite).tolist()
        else:
            self.genepool = elite if elite is not None else []

        if parents is not None:
            moms, dads = self.tournament(parents)
            for i in range(8):
                self.genepool.append(net.Net(gen, moms[i], dads[i]))
        else:
            current = len(self.genepool)
            for i in range(10 - current):
                self.genepool.append(net.Net(gen))

    @staticmethod
    def tournament(parents):
        """Tournament selection to generate pairs of parents.

        The net with the highest score is selected to be a parent with probability p, then second highest p*(p-1), third
        highest p*(p-1)**2, etc.

        Parameters
        ----------
        parents : List[Net]
            Nets from which the population will be spawned.

        Returns
        -------
        moms : List[Net]
            A list of 8 nets.
        dads : List[Net]
            A list of 8 nets.
        """
        moms = []
        dads = []
        while len(moms) < 8:
            m1, m2, d1, d2 = np.random.choice(parents, 4)
            if m1.score > m2.score:
                moms.append(m1)
            else:
                moms.append(m2)
            if d1.score > d2.score:
                dads.append(d1)
            else:
                dads.append(d2)
        return moms, dads

    def play_games(self, games=50):
        """Get each net in the population to play a certain number of games.

        Parameters
        ----------
        games : int
            The number of games each net should play.
        """
        for n in self.genepool:
            n.learn_game(games=games)

    def save_generation(self):
        """Write the top 2 nets from every 20th generation to file. Assumes list is sorted by score.

        Save file from 20 generations ago will be deleted if it exists.
        """
        if not self.generation % 20 and self.generation != 0:
            name = 'Generation' + str(self.generation)
            np.save(name, self.genepool[:2])
            old_name = 'Generation' + str(self.generation-20) + '.npy'
            if isfile(old_name):
                remove(old_name)

    def sort_by_score(self):
        """Sort the genepool in descending order by each net's score."""
        self.genepool.sort(key=lambda n: n.score, reverse=True)


def train_population(final_gen, initial_gen=0, elite=None):
    """Run a micro-genetic algorithm to evolve a good neural net.

    Each population defaults to 10 nets and plays 50 games. The top 2 from each generation are copied to the next one,
    but all have the opportunity to reproduce. Every 10 generations, all but the top 2 are killed and 8 new nets are
    randomly generated to add to the 2 that survived.

    Parameters
    ----------
    final_gen : int
        The total number of generations played is final_gen - initial_gen.  Recommended to be 10*n for positive integer
        values of n.
    initial_gen : int
        The total number of generations played is final_gen - initial_gen
    elite : List[int]
        List of nets to copy directly into the new population.

    Returns
    -------
    top_scores : List[float]
        List of floats. The top score in each generation.
    best_net : Net
        The trained net that performs best.
    """
    parents = None
    top_scores = []

    for gen in range(initial_gen+1, final_gen+1):
        pop = Population(gen, elite, parents)

        print('Playing games for generation', gen, 'of', final_gen)
        pop.play_games()
        pop.sort_by_score()
        pop.save_generation()

        top_scores.append(pop.genepool[0].score)

        elite = pop.genepool[:2]

        if not gen % 10:
            parents = []
        else:
            parents = pop.genepool

        print('Best net\'s generation =', pop.genepool[0].generation)
        print('Best net\'s score =', np.rint(pop.genepool[0].score), '\n')

    plt.figure()
    plt.title('log_2(Score)')
    plt.plot(np.log2(top_scores))

    print('Finding best...')
    best_net = find_best(pop)

    filename = "BestNetGen" + str(final_gen)
    np.save(filename, best_net)

    return top_scores, best_net


def find_best(pop):
    """Play 200 games for each net in population. Print stats and best model.

    Parameters
    ----------
    pop : Population
        The population to find the best network in.

    Returns
    -------
    Net
        The best network in pop's genepool.
    """
    pop.play_games(200)
    score = [np.rint(i.score) for i in pop.genepool]
    tile = [np.rint(i.highest_tile) for i in pop.genepool]
    best = np.argmax(score)

    print(score)
    print(tile)
    print('Max score =', np.max(score))
    print('Min score =', np.min(score))
    print('Mean score =', np.mean(score))
    print('Median score =', np.median(score))
    print('Best model =', best)

    return pop.genepool[best]