from game import Action, Game
import numpy as np
from players import OrderedPlayer
import unittest


class TestOrderedPlayer(unittest.TestCase):
    def test_first_move(self):
        g = Game()
        g.board = np.array([[0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        move = OrderedPlayer()._choose_action(g)
        self.assertEqual(move, Action.DOWN)

    def test_move_no_down(self):
        g = Game()
        g.board = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 1, 0]])
        move = OrderedPlayer()._choose_action(g)
        self.assertEqual(move, Action.RIGHT)

    def test_second_move(self):
        g = Game()
        g.board = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0]])
        player = OrderedPlayer()
        player.previous_action = Action.DOWN
        move = player._choose_action(g)
        self.assertEqual(move, Action.RIGHT)

    def test_play_multiple_games(self):
        player = OrderedPlayer()
        player.play_multiple_games(3, progress_bar=False)
        self.assertEqual(player.get_num_games_played(), 3)


if __name__ == '__main__':
    unittest.main()
