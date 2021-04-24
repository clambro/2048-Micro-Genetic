from players import RandomPlayer
import unittest


class TestRandomPlayer(unittest.TestCase):
    def test_play_multiple_games(self):
        player = RandomPlayer()
        player.play_multiple_games(3, progress_bar=False)
        self.assertEqual(player.get_num_games_played(), 3)


if __name__ == '__main__':
    unittest.main()
