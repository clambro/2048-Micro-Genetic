# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:51:04 2017

@author: Costa
"""

import game
import numpy as np


class Net(object):
    """A simple feed-forward neural net to play 2048.

    Layers:
        1. Input layer of 16 nodes corresponding to the 16 tiles on the board.
        2. Fully-connected hidden layer of 16 nodes.
        3. Output layer of 4 nodes corresponding to left, up, right, down keys.

    Attributes:
        score: A float. The net's score geometrically averaged over the games
               it played.
        generation: An integer. Used in the genetic algorithm to track age.
        highest_tile: A float. The geometric mean of the highest tiles in all
                      the games the net played.
        chromosome: A 340x1 numpy array containing the weights for the
                    matrices. (16 inputs; 1 hidden layer of size 16, plus bias.
                    17x16 matrix then 17x4 = 340 elements)
    """

    def __init__(self, gen=0, mom=None, dad=None, chromosome=None):
        """Inits the neural net.

        If a chromosome is given, load that as the net's chromosome.  If
        parents are given, use them to generate a child chromosome.  If nothing
        is given, generate a random chromosome.

        Args:
            gen: An integer. The current generation.
            mom: A net from which the chromosome will be sampled.
            dad: The other net from which the chromosome will be sampled.
            chromosome: A 340x1 numpy array containing the matrix weights.
        """
        self.score = 0
        self.generation = gen
        self.highest_tile = 0
        if chromosome:
            self.chromosome = chromosome
        elif not mom or not dad:
            self.chromosome = np.random.uniform(-0.4, 0.4, 340)
        # Build chromosome by taking elements from mom and dad, biased 60/40
        # towards higher-scoring parent. Then add random mutations.
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
                             else 0 for i in range(len(self.chromosome))])
        self.chromosome += mutation

    def relu(self, x):
        """Leaky ReLU"""
        return np.maximum(0.01*x, x)

    def make_move(self, board):
        """Input board into net and feed-forward to get a move direction.

        Returns:
            moves: A list of integers corresponding to movement directions,
                   sorted according to the net's output.
        """
        # Normalize board.  Only relative tile magnitude matters.
        x = board.reshape(16) / np.max(board)
        x = np.append(1, board)  # Add bias
        W_xh = self.chromosome[:272].reshape((17, 16))
        W_hy = self.chromosome[272:].reshape((17, 4))
        h = self.relu(x @ W_xh)
        h = np.append(1, h)  # Add bias
        y = h @ W_hy  # No non-linearity needed. We only care about order.
        moves = np.array([37, 38, 39, 40])  # [left, up, right, down]
        return moves[y.argsort()[::-1]]

    def learn_game(self, games):
        """Net plays games and calculates average scores and highest tiles.

        Args:
            games: Integer. The number of games to average over.
        """
        self.score = 1
        self.highest_tile = 1
        for i in range(games):
            g = game.Game()
            while not g.game_over:
                keys = self.make_move(g.board)
                board_copy = np.copy(g.board)
                score_copy = np.copy(g.score)
                for key in keys:
                    g.process_key(key)
                    if not g.last_move_illegal:
                        break
                    else:
                        g.last_move_illegal = False
                        g.board = board_copy
                        g.score = score_copy.tolist()
            # Score and highest tile are geometric averages.
            self.score *= g.score**(1/games)
            self.highest_tile *= g.highest_tile**(1/games)

    def play_game(self, show=True):
        """Net plays one game of 2048 with optional fancy graphics.

        Args:
            show: If True, Python will display an animated plot of the board.
        """
        g = game.Game()
        if show:
            ax = g.display_game()
        while not g.game_over:
            keys = self.make_move(g.board)
            board_copy = np.copy(g.board)
            score_copy = np.copy(g.score)
            for key in keys:
                g.process_key(key)
                if not g.last_move_illegal:
                    break
                else:
                    g.last_move_illegal = False
                    g.board = board_copy
                    g.score = score_copy.tolist()
            if show:
                ax = g.display_game(ax)
        self.score = g.score
        self.highest_tile = g.highest_tile

    def get_stats(self, games=10000):
        """Play games and analyze the net's highest tiles and average score.

        Returns:
            scores: A list containing the score in each game
            tiles: A list containing the highest tile in each game
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

    Returns:
        scores: A list containing the score in each game
        tiles: A list containing the highest tile in each game
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