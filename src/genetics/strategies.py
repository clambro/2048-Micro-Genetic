from genetics.population import Population
import matplotlib.pyplot as plt
import numpy as np


NETS_PER_POP = 32
NUM_ELITE = 1


def run_micro_genetic_alg(num_generations, pop=None):
    """Run a micro-genetic algorithm to evolve a good neural network.

    Each network plays 20 games and the weakest half are removed from the population. Then 30 more games are played and
    the weakest half are again removed. Finally, for each remaining network whose average score is in range of the
    elite network's lower bound, 250 more games are played. The top networks then go on to populate the next generation.
    Every 30 generations, all non-elite networks are randomized to improve diversity.

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
        pop = Population(NETS_PER_POP, NUM_ELITE, pop)
        if not gen % 30 and gen > 0:
            print('Randomizing non-elite networks to improve diversity.')
            pop.randomize()

        print(f'Playing games for generation {pop.generation} ({gen + 1} of {num_generations})')

        print('Playing first 20 games.')
        pop.play_games(20, include_elites=False)
        num_to_filter = NETS_PER_POP // 2 - len(pop.elites)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:num_to_filter]

        print('Playing next 30 games.')
        pop.play_games(30, include_elites=False)
        num_to_filter = NETS_PER_POP // 4 - len(pop.elites)
        pop.networks = pop.get_sorted_networks(include_elites=False)[:num_to_filter]

        if not pop.elites:
            print('Playing final 250 games to determine elites.')
            pop.play_games(250, include_elites=False)
        else:
            elite = pop.elites[0]
            log_st_err = np.std(np.log(elite.scores)) / np.sqrt(elite.get_num_games_played())
            thresh = elite.get_avg_score() / np.exp(2 * log_st_err)  # Approximate lower bound of score estimate.
            print(f'Playing 250 games for networks above {np.rint(thresh)}.')
            pop.play_games(250, include_elites=False, thresh=thresh)

        if not pop.generation % 10 and pop.generation != 0:
            pop.save(f'Generation{pop.generation}.pkl')

        top_network = pop.get_sorted_networks(include_elites=True)[0]
        top_scores.append(top_network.get_avg_score())

        print('Best network\'s generation =', top_network.generation)
        print('Best network\'s score =', np.rint(top_scores[-1]))
        print('Best network\'s highest tile =', np.rint(top_network.get_avg_highest_tile()), '\n')

    plt.figure()
    plt.title('Network Improvement vs Generation')
    plt.xlabel('Log Generation')
    plt.ylabel('Log Highest Score')
    plt.loglog(top_scores)
    plt.savefig('scores_per_generation.png')

    pop.save(f'Generation{pop.generation}.pkl')

    return top_scores, top_network
