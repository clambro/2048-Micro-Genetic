from game.action import Action, DIRECTIONS
import numpy as np


class Board:
    """

    """

    def __init__(self):
        """"""
        self.board = np.zeros((4, 4), dtype=np.int)
        self.game_over = False
        self.add_tile()
        self.add_tile()

    def move(self, direction):
        """Execute a move in the given direction.

        Rotate the board so that key points leftwards, move left, then rotate back to the original position.

        Parameters
        ----------
        direction : Action
            The ordinal value of the key pressed.
        """
        if self.game_over:
            return False, 0

        prev_board = np.copy(self.board)
        if direction == Action.LEFT:
            rot = 0
        elif direction == Action.UP:
            rot = 1
        elif direction == Action.RIGHT:
            rot = 2
        else:
            rot = -1
        self.board = np.rot90(self.board, rot)
        # Slide, merge, slide pattern reflects the official 2048 rules
        self.slide_left()
        points_earned = self.merge_left()
        self.slide_left()
        self.board = np.rot90(self.board, -1*rot)

        # If the board is unchanged by the move, then it was illegal.
        move_was_legal = not np.array_equal(self.board, prev_board)
        if move_was_legal:
            self.add_tile()
        self.game_over = self.is_game_over()
        return move_was_legal, points_earned

    def add_tile(self):
        """Adds a 2 or 4 tile randomly to the current game board."""
        # In 2048, there is a 10% chance of a 4 being added instead of a 2.
        if np.random.random() > 0.9:
            val = 2
        else:
            val = 1
        board = self.board.reshape(16)
        valid_pos = [i for i in range(16) if not board[i]]
        pos = np.random.choice(valid_pos)
        board[pos] = val
        self.board = board.reshape((4, 4))

    def has_legal_moves(self):
        """Return True if there are no legal moves, else False."""
        orig_board = np.copy(self.board)
        for direction in DIRECTIONS:
            move_was_legal, _ = self.move(direction)
            self.board = orig_board
            if move_was_legal:
                return True
        return False

    def is_game_over(self):
        """"""
        return np.all(self.board) and ~self.has_legal_moves()

    def slide_left(self):
        """Slides tiles left one column at a time, but doesn't merge them."""
        for row in range(4):
            new_row = [i for i in self.board[row, :] if i != 0]
            new_row = new_row + [0] * (4 - len(new_row))
            self.board[row, :] = np.array(new_row)

    def merge_left(self):
        """Merge tiles and increase score according to 2048 rules."""
        points_earned = 0
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j+1] != 0:
                    self.board[i, j] = self.board[i, j] + 1
                    self.board[i, j+1] = 0
                    points_earned += 2**self.board[i, j]
        return points_earned
