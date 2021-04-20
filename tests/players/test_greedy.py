from game.action import Action
from game.game import Game
import numpy as np
from players.greedy import GreedyPlayer
import unittest


class TestGreedyPlayer(unittest.TestCase):
    def setUp(self):
        self.player = GreedyPlayer()
        self.game = Game()

    def test_greedy_action_trivial(self):
        self.game.board = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 2],
                                    [0, 0, 0, 2]])
        self.assertIn(self.player._choose_action(self.game), [Action.UP, Action.DOWN])

    def test_greedy_action_choice(self):
        self.game.board = np.array([[4, 4, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 2],
                                    [0, 0, 0, 2]])
        self.assertEqual(self.player._choose_action(self.game), Action.LEFT)

    def test_greedy_action_multiple(self):
        self.game.board = np.array([[3, 3, 0, 0],
                                    [0, 0, 0, 0],
                                    [1, 2, 1, 2],
                                    [1, 2, 1, 2]])
        self.assertIn(self.player._choose_action(self.game), [Action.UP, Action.DOWN])

    def test_play_multiple_games(self):
        self.player.play_multiple_games(3, progress_bar=False)
        self.assertEqual(self.player.get_num_games_played(), 3)


if __name__ == '__main__':
    unittest.main()
