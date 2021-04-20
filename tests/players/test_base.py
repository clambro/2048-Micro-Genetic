import numpy as np
from players.base import Player
import unittest
from unittest.mock import patch


class TestPlayer(unittest.TestCase):
    @patch('players.base.Player.__abstractmethods__', set())
    def setUp(self):
        """Instantiates an abstract player class by removing its abstract methods through a patch."""
        self.player = Player()

    def test_get_avg_score_no_games(self):
        np.testing.assert_equal(self.player.get_avg_score(), np.nan)  # This function treats NaNs as equal.

    def test_get_avg_score(self):
        self.player.scores = [10, 100, 1000]
        self.assertAlmostEqual(self.player.get_avg_score(), 100)

    def test_get_avg_highest_tile_no_games(self):
        np.testing.assert_equal(self.player.get_avg_highest_tile(), np.nan)  # This function treats NaNs as equal.

    def test_get_avg_highest_tile(self):
        self.player.highest_tiles = [10, 100, 1000]
        self.assertAlmostEqual(self.player.get_avg_highest_tile(), 100)

    def test_get_num_games(self):
        self.assertEqual(self.player.get_num_games_played(), 0)
        self.player.scores = [10, 100, 1000]
        self.assertEqual(self.player.get_num_games_played(), 3)


if __name__ == '__main__':
    unittest.main()
