from game.action import DIRECTIONS
import numpy as np
from players.base import Player


class Net(Player):
    """A simple feed-forward neural net to play 2048.

    Layers:
        1. Input layer of 16 nodes corresponding to the 16 tiles on the board.
        2. Fully-connected hidden layer of 16 nodes.
        3. Output layer of 4 nodes corresponding to left, up, right, down keys.

    Attributes
    ----------
    generation : int
        Which generation the network belongs to.
    chromosome : ndarray
        A 340x1 numpy array containing the weights for the matrices. (16 inputs; 1 hidden layer of size 16, plus bias.
        17x16 matrix then 17x4 = 340 elements)
    """

    def __init__(self, gen=0, mom=None, dad=None, chromosome=None):
        """Builds the network from a chromosome if given, or two parents, falling back to random generation if neither.

        Parameters
        ----------
        gen : int
            The current generation.
        mom : Optional[Net]
            A net from which the chromosome will be sampled.
        dad : Optional[Net]
            The other net from which the chromosome will be sampled.
        chromosome : Optional[ndarray]
            A 340x1 numpy array containing the network weights.
        """
        super().__init__()
        self.generation = gen
        if chromosome:
            self.chromosome = chromosome
        elif not mom or not dad:
            self.chromosome = np.random.uniform(-0.4, 0.4, 340)
        elif mom.score > dad.score:
            self.chromosome = np.array([
                    mom.chromosome[i] if np.random.random() > 0.4
                    else dad.chromosome[i]
                    for i in range(len(mom.chromosome))
                    ])
            self.mutate()
        else:
            self.chromosome = np.array([
                    dad.chromosome[i] if np.random.random() > 0.4
                    else mom.chromosome[i]
                    for i in range(len(mom.chromosome))
                    ])
            self.mutate()

    def mutate(self):
        """Add random mutations to 2% of net's chromosome."""
        mutation = np.array([np.random.randn()/10 if np.random.random() < 0.02
                             else 0 for _ in range(len(self.chromosome))])
        self.chromosome += mutation

    @staticmethod
    def relu(x):
        """Leaky ReLU"""
        return np.maximum(0.01*x, x)

    def make_move(self, board):
        """Input board into net and feed-forward to get a move direction.

        Parameters
        ----------
        board : ndarray
            The board state to calculate the move for.

        Returns
        -------
        moves : ndarray
            A list of integers corresponding to movement directions, sorted according to the net's output.
        """
        x = board.reshape(16) / np.max(board)  # Only relative tile magnitude matters.
        x = np.append(1, x)  # Add bias
        w_xh = self.chromosome[:272].reshape((17, 16))
        w_hy = self.chromosome[272:].reshape((17, 4))
        h = self.relu(x @ w_xh)
        h = np.append(1, h)  # Add bias
        return (h @ w_hy).tolist()  # No non-linearity needed. We only care about order.

    def choose_action(self, game):
        """"""
        legal_moves = game.get_legal_moves()
        best_move = None
        highest_priority = -np.inf
        for direction, priority in zip(DIRECTIONS, self.make_move(game.board)):
            if direction in legal_moves and priority > highest_priority:
                best_move = direction
                highest_priority = highest_priority
        return best_move

    def play_multiple_games(self, games):
        """Play games and calculate (geometric) average scores and highest tiles.

        Parameters
        ----------
        games: int
            The number of games to play.
        """
        for _ in range(games):
            super().play_game(False)

    def get_stats(self, games=10000):
        """Play games and analyze the net's highest tiles and average score.

        Parameters
        ----------
        games : int
            The number of games to play.

        Returns
        -------
        scores : List[int]
            The score in each game
        tiles : List[int]
            The highest tile in each game
        """
        dic = {}
        scores = []
        tiles = []
        for i in range(games):
            if not i % 100:
                print('Playing game number', i)
            self.play_game(False)
            # Update tile count in dictionary
            tile = self.highest_tile
            if tile in dic:
                dic[tile] = dic[tile] + 1
            else:
                dic[tile] = 1
            # Take geometric mean of score and tile
            scores.append(self.score)
            tiles.append(self.highest_tile)
        for key, value in sorted(dic.items(), key=lambda x: x[0]):
            print(key, ':', np.round(100*value/games, 1), '%')
        print('Average Score =', np.rint(np.exp(np.mean(np.log(scores)))))
        return scores, tiles


def test_random():
    """Play one game for each of 10000 nets and get summary statistics.

    Returns
    -------
    scores : List[int]
        The score in each game
    tiles : List[int]
        The highest tile in each game
    """
    dic = {}
    scores = []
    tiles = []
    for i in range(10000):
        n = Net()
        if not i % 100:
            print('Playing game number', i)
        n.play_game(False)
        # Update tile count in dictionary
        tile = n.highest_tile
        if tile in dic:
            dic[tile] = dic[tile] + 1
        else:
            dic[tile] = 1
        # Take geometric mean of score and tile
        scores.append(n.score)
        tiles.append(n.highest_tile)
    for key, value in sorted(dic.items(), key=lambda x: x[0]):
        print(key, ':', np.round(value/100, 2), '%')
    print('Average Score =', np.rint(np.exp(np.mean(np.log(scores)))))
    print('Max Score =', max(scores))
    print('Min Score =', min(scores))
    return scores, tiles
