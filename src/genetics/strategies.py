from genetics.population import Population, NETS_PER_POP, NUM_ELITE
import matplotlib.pyplot as plt
import numpy as np


def run_micro_genetic_alg(num_generations, pop=None):
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
    top_scores = []
    top_network = None
    for gen in range(num_generations):
        pop = Population(pop)
        print(f'Playing games for generation {pop.generation} ({gen + 1} of {num_generations})')

        print('Playing first 10 games.')
        pop.play_games(10, include_elites=False)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:NETS_PER_POP // 2 - NUM_ELITE]

        print('Playing next 40 games.')
        pop.play_games(40, include_elites=False)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:NETS_PER_POP // 4 - NUM_ELITE]

        print('Playing final 150 games.')
        pop.play_games(150, include_elites=False)

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
