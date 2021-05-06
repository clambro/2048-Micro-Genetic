from copy import copy
import pickle
from players import NetworkPlayer
import numpy as np
from tqdm import tqdm


class Population:
    """A collection of networks.

    Attributes
    ----------
    generation : int
        The current generation.
    num_nets : int
        The total number of networks in the population on creation.
    num_elite : int
        The number of elite networks to pass directly to the next population.
    networks : List[NetworkPlayer]
        The networks in the population.
    elites : List[NetworkPlayer]
        Additional elite networks from a previous generation that may be treated differently.
    similarity : float
        The average similarity (overlapping weights) between all networks in the population.
    """

    def __init__(self, num_nets=None, num_elite=None, pop=None):
        """Builds the population by either reproducing from a previous one or randomly generating networks.

        Parameters
        ----------
        num_nets : int
            The total number of networks in the population on creation.
        num_elite : int
            The number of elite networks to pass directly to the next population.
        pop : Optional[Union[Population, str]]
            The population from which to spawn this population, or a path leading to it. If None, the population will
            be generated randomly. Overwrites the given num_net and num_elite parameters to match it.
        """
        if pop is None:
            if None in [num_nets, num_elite]:
                raise ValueError('If pop is none, then both num_nets and num_elite must be given.')
            self.generation = 1
            self.num_nets = num_nets
            self.num_elite = num_elite
            self.elites = []
            self.networks = [NetworkPlayer(gen=1) for _ in range(self.num_nets)]
        else:
            if isinstance(pop, str):
                with open(pop, 'rb') as f:
                    pop = pickle.load(f)
            self.generation = pop.generation + 1
            self.num_nets = pop.num_nets
            self.num_elite = pop.num_elite
            prev_networks = pop.get_sorted_networks(include_elites=True)
            self.elites = prev_networks[:self.num_elite]
            self.networks = self._spawn_children(prev_networks)
        self.similarity = self._determine_similarity()

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
        parents = [np.random.choice(parents, 2, replace=False) for _ in range(self.num_nets - self.num_elite)]
        return [NetworkPlayer(gen=self.generation, mom=p[0], dad=p[1]) for p in parents]

    def _determine_similarity(self):
        """Determine the mean similarity between all pairs of networks in the population.

        Returns
        -------
        float
            The mean similarity for the population as a number between 0 and 1.
        """
        networks = self.networks + self.elites
        similarity = []
        for i, n1 in enumerate(networks):
            for n2 in networks[i+1:]:
                similarity.append(n1.calculate_similarity(n2))
        return np.mean(similarity)

    def randomize_population(self):
        """Randomize the non-elite networks without changing the total number and recalculate the similarity."""
        self.networks = [NetworkPlayer() for _ in self.networks]
        self.similarity = self._determine_similarity()

    def play_games(self, games, include_elites, progress_bar=True, thresh=0):
        """Get each network in the population to play a certain number of games.

        Parameters
        ----------
        games : int
            The number of games each network should play.
        include_elites : bool
            Whether or not to make the elites play as well.
        progress_bar : bool
            Whether or not to display a tqdm progress bar.
        thresh : float
            Only networks with an average score above this threshold will play games.
        """
        networks = copy(self.networks)
        if include_elites:
            networks += self.elites
        networks = [n for n in networks if not n.scores or n.get_avg_score() > thresh]
        if progress_bar:
            iterator = tqdm(networks)
        else:
            iterator = networks
        for n in iterator:
            n.play_multiple_games(games, progress_bar=False)

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
        networks = copy(self.networks)
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
