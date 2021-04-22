from players.manual import ManualPlayer
import unittest


class TestManualPlayer(unittest.TestCase):
    def test_play_game_no_display(self):
        with self.assertRaises(ValueError) as e:
            ManualPlayer().play_game(display=False)
        self.assertIn('You need to display the board to play a manual game!', str(e.exception))

    def test_play_multiple_games(self):
        with self.assertRaises(TypeError) as e:
            ManualPlayer().play_multiple_games(5)
        self.assertIn('Multiple games without graphics cannot be played manually.', str(e.exception))


if __name__ == '__main__':
    unittest.main()
