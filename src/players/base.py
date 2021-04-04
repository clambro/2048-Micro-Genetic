from abc import ABC, abstractmethod
from game.action import Action
from game.game import Game
import matplotlib.pyplot as plt
import numpy as np


class Player(ABC):
    """

    """

    def __init__(self):
        """"""
        self.scores = []
        self.highest_tiles = []

    def get_avg_score(self):
        """"""
        return np.exp(np.mean(np.log(self.scores)))

    def get_avg_highest_tile(self):
        """"""
        return np.exp(np.mean(np.log(self.highest_tiles)))

    def get_num_games_played(self):
        """"""
        return len(self.scores)

    def play_game(self, display):
        """"""
        game = Game()
        if display:
            ax = game.display_board()
        while not game.game_over:
            action = self.choose_action(game)
            if action == Action.QUIT:
                break
            game.move(action)
            if display:
                game.display_board(ax)
        if display:
            plt.close()
            print(f'Game Over. Final score was {game.score}. Highest tile was {game.highest_tile}.')
        self.scores.append(game.score)
        self.highest_tiles.append(game.highest_tile)

    @abstractmethod
    def choose_action(self, game):
        """"""
        pass
