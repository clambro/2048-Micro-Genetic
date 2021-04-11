from genetics.population import Population, NETS_PER_POP, NUM_ELITE
import matplotlib.pyplot as plt
import numpy as np


def run_micro_genetic_alg(num_generations, pop=None):
    """Run a micro-genetic algorithm to evolve a good neural network.

    Each network plays 20 games, and the lowest half are removed from the population. Then 30 more games are played and
    the lowest half are again removed. Finally, 250 more games are played to determine the true average score for each
    remaining network. These go on to populate the next generation.

    Parameters
    ----------
    num_generations : int
        The total number of generations to run.
    pop : Optional[Population]
        Starting population. If None, one will be randomly generated.

    Returns
    -------
    top_scores : List[float]
        List of floats. The top average score in each generation.
    best_net : NetworkPlayer
        The trained networks that performs best.
    """
    top_scores = []
    top_network = None
    for gen in range(num_generations):
        pop = Population(pop)
        print(f'Playing games for generation {pop.generation} ({gen + 1} of {num_generations})')

        print('Playing first 20 games.')
        pop.play_games(20, include_elites=False)
        num_to_filter = NETS_PER_POP // 2 - len(pop.elites)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:num_to_filter]

        print('Playing next 30 games.')
        pop.play_games(30, include_elites=False)
        num_to_filter = NETS_PER_POP // 4 - len(pop.elites)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:num_to_filter]

        print('Playing final 250 games.')
        pop.play_games(250, include_elites=False)

        if not pop.generation % 10 and pop.generation != 0:
            pop.save(f'Generation{pop.generation}.pkl')

        top_network = pop.get_sorted_networks(include_elites=True)[0]
        top_scores.append(top_network.get_avg_score())

        print('Best network\'s generation =', top_network.generation)
        print('Best network\'s score =', np.rint(top_scores[-1]))
        print('Best network\'s highest tile =', np.rint(top_network.get_avg_highest_tile()), '\n')

    plt.figure()
    plt.title('log_2(Highest Score)')
    plt.plot(np.log2(top_scores))

    pop.save(f'Generation{pop.generation}.pkl')

    return top_scores, top_network
