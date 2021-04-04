from game.action import Action, DIRECTIONS
import matplotlib.pyplot as plt
import numpy as np


class Game:
    """A single game of 2048.

    A game consists of a 4x4 board of tiles with integer values representing the log2 of the tile's value. The tiles
    can be shifted up, down, left, or right, and when two tiles of the same value touch, they merge and their value is
    doubled. If the board is full and there are no legal moves, the game ends.

    Attributes
    ----------
    board : ndarray
        A 4x4 integer array of the log2 tile value at each board position.
    score : int
        The current score.
    highest_tile : int
        The highest-value tile on the board.
    game_over : bool
        Whether or not the game is over.
    """

    def __init__(self):
        """Sets up the board and clears the key-state."""
        self.board = np.zeros((4, 4), dtype=np.int)
        self.add_tile()
        self.add_tile()
        self.score = 0
        self.highest_tile = 0
        self.game_over = False

    def add_tile(self):
        """Adds a 2 or 4 tile randomly to the current game board and checks for game over."""
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
        self.highest_tile = 2 ** np.max(self.board)
        if self.board.all() and not self.get_legal_moves():
            self.game_over = True

    def get_legal_moves(self):
        """Return True if there are no legal moves, else False."""
        if self.game_over:
            return []
        orig_board = np.copy(self.board)
        orig_score = int(np.copy(self.score))
        legal_moves = []
        for direction in DIRECTIONS:
            move_was_legal = self.move(direction, add_tile=False)
            if move_was_legal:
                # Reset parameters that may have changed during the move.
                self.board = np.copy(orig_board)
                self.score = int(np.copy(orig_score))
                legal_moves.append(direction)
        return legal_moves

    def slide_left(self):
        """Slides tiles left one column at a time, but doesn't merge them."""
        for row in range(4):
            new_row = [i for i in self.board[row, :] if i != 0]
            new_row = new_row + [0] * (4 - len(new_row))
            self.board[row, :] = np.array(new_row)

    def merge_left(self):
        """Merge tiles and increase score according to 2048 rules."""
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j+1] != 0:
                    self.board[i, j] = self.board[i, j] + 1
                    self.board[i, j+1] = 0
                    self.score += 2**self.board[i, j]

    def move(self, direction, add_tile=True):
        """Execute a move in the given direction.

        Rotate the board so that key points leftwards, move left, then rotate back to the original position.

        Parameters
        ----------
        direction : Action
            The ordinal value of the key pressed.
        """
        if self.game_over:
            return False

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
        self.merge_left()
        self.slide_left()
        self.board = np.rot90(self.board, -1 * rot)

        # If the board is unchanged by the move, then it was illegal.
        move_was_legal = not np.array_equal(self.board, prev_board)
        if move_was_legal and add_tile:
            self.add_tile()
        return move_was_legal

    def display_board(self, axes=None):
        """Display the board as a matplotlib figure.

        Parameters
        ----------
        axes : Optional[Axes]
            Used to plot over previous figure, otherwise a new Axes object is generated.

        Returns
        -------
        axes : Axes
            The current figure's Axes.
        """
        if axes is None:
            fig, axes = plt.subplots()
            show_flag = True
            plt.pause(1)  # Gives you a bit of time to adjust the plot
        else:
            show_flag = False
            axes.clear()
        axes.set_title('Score = ' + str(self.score))
        plt.imshow(self.board, cmap='summer')
        for (j, i), label in np.ndenumerate(2**self.board):
            if label != 1:
                axes.text(i, j, label, ha='center', va='center')
        plt.pause(0.001)  # Need a slight delay or the plot won't render
        if show_flag:
            plt.show()
        return axes
