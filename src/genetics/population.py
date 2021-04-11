import pickle
from players.net import NetworkPlayer
import numpy as np
from tqdm import tqdm


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
        pop : Optional[Union[Population, str]]
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
            prev_networks = pop.get_sorted_networks(include_elites=True)
            self.elites = prev_networks[:NUM_ELITE]
            self.networks = self._spawn_children(prev_networks)

    def _spawn_children(self, parents):
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
        return [NetworkPlayer(gen=self.generation, mom=m, dad=d) for m, d in zip(moms, dads)]

    def play_games(self, games, include_elites, progress_bar=True):
        """Get each network in the population to play a certain number of games.

        Parameters
        ----------
        games : int
            The number of games each network should play.
        include_elites : bool
            Whether or not to make the elites play as well.
        progress_bar : bool
            Whether or not to display a tqdm progress bar.
        """
        networks = self.networks
        if include_elites:
            networks += self.elites
        if progress_bar:
            iterator = tqdm(networks)
        else:
            iterator = networks
        for n in iterator:
            n.play_multiple_games(games)

    def get_sorted_networks(self, include_elites):
        """Sort the population's networks in descending order by each network's average score.

        Parameters
        ----------
        include_elites : bool
            Whether or not to include the elites in the result.

        Returns
        -------
        networks : List[NetworkPlayer]
            The networks sorted by average score.
        """
        networks = self.networks
        if include_elites:
            networks += self.elites
        networks.sort(key=lambda n: n.get_avg_score(), reverse=True)
        return networks

    def save(self, path):
        """Save the population to a file.

        Parameters
        ----------
        path : str
            The save path.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
