from players.network import NetworkPlayer
import unittest


class TestNetworkPlayer(unittest.TestCase):
    def test_similarity(self):
        player = NetworkPlayer()
        self.assertEqual(player.calculate_similarity(player), 1)

    def test_similarity_and_child(self):
        player1 = NetworkPlayer()
        player2 = NetworkPlayer()
        child = NetworkPlayer(mom=player1, dad=player2)
        self.assertGreater(child.calculate_similarity(player1), 0.2)
        self.assertGreater(child.calculate_similarity(player2), 0.2)

    def test_play_multiple_games(self):
        player = NetworkPlayer()
        player.play_multiple_games(3, progress_bar=False)
        self.assertEqual(player.get_num_games_played(), 3)


if __name__ == '__main__':
    unittest.main()
