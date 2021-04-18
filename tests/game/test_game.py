from game.action import Action, DIRECTIONS
from game.game import Game
import numpy as np
import unittest


class TestGame(unittest.TestCase):
    def test_legal_start(self):
        g = Game()
        self.assertFalse(g.game_over)
        self.assertEqual(g.score, 0)
        self.assertIn(g.highest_tile, [2, 4])  # Game could start with a 4 tile.
        self.assertIn(np.sum(g.board), [2, 3, 4])
        self.assertEqual(np.sum(g.board > 0), 2)

    def test_legal_moves(self):
        g = Game()
        g.board = np.array([[0, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 0]])
        self.assertSetEqual(set(g.get_legal_moves()), set(DIRECTIONS))

        g.board = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 0]])
        self.assertSetEqual(set(g.get_legal_moves()), {Action.LEFT, Action.UP, Action.RIGHT})

        g.board = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        self.assertSetEqual(set(g.get_legal_moves()), {Action.LEFT, Action.DOWN, Action.RIGHT})

        g.board = np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        self.assertSetEqual(set(g.get_legal_moves()), {Action.DOWN, Action.UP, Action.RIGHT})

        g.board = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])
        self.assertSetEqual(set(g.get_legal_moves()), {Action.LEFT, Action.UP, Action.DOWN})

        g.board = np.array([[0, 0, 0, 0],
                            [0, 0, 1, 2],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1]])
        self.assertSetEqual(set(g.get_legal_moves()), {Action.LEFT, Action.UP, Action.DOWN})

    def test_legal_moves_game_over(self):
        g = Game()
        g.game_over = True
        self.assertListEqual(g.get_legal_moves(), [])

    def test_legal_moves_game_state_unchanged(self):
        g = Game()
        board = 4 * np.ones((4, 4))
        g.board = board
        g.get_legal_moves()

        self.assertFalse(g.game_over)
        self.assertEqual(g.score, 0)
        self.assertIn(g.highest_tile, [2, 4])

    @staticmethod
    def set_up_for_move_test(direction):
        """Helper function to set up a board for testing directional moves"""
        g = Game()
        g.board = np.array([[0, 1, 0, 0],
                            [2, 2, 0, 2],
                            [1, 2, 0, 0],
                            [1, 0, 1, 0]])
        np.random.seed(2112)
        g.move(direction)
        return g

    def test_move_left(self):
        g = self.set_up_for_move_test(Action.LEFT)
        correct_board = np.array([[1, 0, 0, 0],
                                  [3, 2, 0, 0],
                                  [1, 2, 0, 1],  # New tile added here based on seed.
                                  [2, 0, 0, 0]])
        np.testing.assert_array_equal(g.board, correct_board)
        self.assertEqual(g.highest_tile, 8)
        self.assertEqual(g.score, 12)
        self.assertFalse(g.game_over)

    def test_move_right(self):
        g = self.set_up_for_move_test(Action.RIGHT)
        correct_board = np.array([[0, 0, 0, 1],
                                  [0, 0, 2, 3],
                                  [0, 1, 1, 2],  # New tile added here based on seed.
                                  [0, 0, 0, 2]])
        np.testing.assert_array_equal(g.board, correct_board)
        self.assertEqual(g.highest_tile, 8)
        self.assertEqual(g.score, 12)
        self.assertFalse(g.game_over)

    def test_move_up(self):
        g = self.set_up_for_move_test(Action.UP)
        correct_board = np.array([[2, 1, 1, 2],
                                  [2, 3, 0, 0],
                                  [0, 0, 0, 0],
                                  [1, 0, 0, 0]])  # New tile added here based on seed.
        np.testing.assert_array_equal(g.board, correct_board)
        self.assertEqual(g.highest_tile, 8)
        self.assertEqual(g.score, 12)
        self.assertFalse(g.game_over)

    def test_move_down(self):
        g = self.set_up_for_move_test(Action.DOWN)
        correct_board = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 0],  # New tile added here based on seed.
                                  [2, 1, 0, 0],
                                  [2, 3, 1, 2]])
        np.testing.assert_array_equal(g.board, correct_board)
        self.assertEqual(g.highest_tile, 8)
        self.assertEqual(g.score, 12)
        self.assertFalse(g.game_over)

    def test_move_illegal(self):
        g = Game()
        board = np.arange(16).reshape(4, 4) + 1  # Filled board has no legal moves.
        g.board = np.copy(board)
        g.highest_tile = 0
        [g.move(d) for d in DIRECTIONS]  # Illegal directions should not change the game state
        np.testing.assert_array_equal(g.board, board)
        self.assertEqual(g.highest_tile, 0)
        self.assertEqual(g.score, 0)
        self.assertFalse(g.game_over)

    def test_move_after_game_over(self):
        g = Game()
        board = np.copy(g.board)
        g.highest_tile = 0
        g.game_over = True
        [g.move(d) for d in DIRECTIONS]  # Cannot move if the game is over.
        np.testing.assert_array_equal(g.board, board)
        self.assertEqual(g.highest_tile, 0)
        self.assertEqual(g.score, 0)
        self.assertTrue(g.game_over)

    def test_move_to_game_over(self):
        g = Game()
        g.board = np.arange(16).reshape(4, 4)  # Guaranteed game over after moving left.
        g.move(Action.LEFT)
        self.assertEqual(g.highest_tile, 2**15)
        self.assertEqual(g.score, 0)
        self.assertTrue(g.game_over)


if __name__ == '__main__':
    unittest.main()
