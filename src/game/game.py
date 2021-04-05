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
        """Sets up the game state with two random tiles."""
        self.board = np.zeros((4, 4), dtype=np.int)
        self._add_tile()
        self._add_tile()
        self.score = 0
        self.highest_tile = 2 ** np.max(self.board)
        self.game_over = False

    def get_legal_moves(self):
        """Determine the legal moves in the current game state.

        Returns
        -------
        List[Action]
            The legal actions that can be taken (not counting quitting).
        """
        if self.game_over:
            return []
        legal_moves = []
        for direction in DIRECTIONS:
            move_was_legal, _, _ = self._move(direction)
            if move_was_legal:
                legal_moves.append(direction)
        return legal_moves

    def move(self, direction):
        """Execute a move in the given direction and update the game state if it was legal.

        Parameters
        ----------
        direction : Action
            The direction to slide the board.
        """
        move_was_legal, new_board, points_earned = self._move(direction)
        if move_was_legal:
            self.board = new_board
            self._add_tile()
            self.highest_tile = 2 ** np.max(self.board)
            self.score += points_earned
            if self.board.all() and not self.get_legal_moves():
                self.game_over = True

    def _move(self, direction):
        """Simulates a move and determines its legality and points earned. Does not update the game state.

        Rotate the board so that direction points leftwards, move left, then rotate back to the original position.

        Parameters
        ----------
        direction : Action
            The direction to slide the board in.

        Returns
        -------
        move_was_legal : bool
            Whether or not the move was legal.
        new_board : ndarray
            The board state after the move. (Identical to self.board for illegal moves).
        points_earned : int
            The points earned by executing the move.
        """
        if self.game_over:
            return False, self.board, 0

        new_board = np.copy(self.board)
        if direction == Action.LEFT:
            rot = 0
        elif direction == Action.UP:
            rot = 1
        elif direction == Action.RIGHT:
            rot = 2
        else:
            rot = -1
        new_board = np.rot90(new_board, rot)
        # Slide, merge, slide pattern reflects the official 2048 rules
        self._slide_left(new_board)
        points_earned = self._merge_left(new_board)
        self._slide_left(new_board)
        new_board = np.rot90(new_board, -1 * rot)

        move_was_legal = not np.array_equal(self.board, new_board)
        return move_was_legal, new_board, points_earned

    @staticmethod
    def _slide_left(board):
        """Slides the tiles of board to the left, in-place, but doesn't merge them.

        Parameters
        ----------
        board : ndarray
            The board to slide in-place.
        """
        for row in range(4):
            new_row = [i for i in board[row, :] if i != 0]
            new_row = new_row + [0] * (4 - len(new_row))
            board[row, :] = np.array(new_row)

    @staticmethod
    def _merge_left(board):
        """Merge the tiles of board to the left, in place.

        Parameters
        ----------
        board : ndarray
            The board to slide in-place.

        Returns
        -------
        points_earned : int
            The points earned during the merge.
        """
        points_earned = 0
        for i in range(4):
            for j in range(3):
                if board[i, j] == board[i, j+1] != 0:
                    board[i, j] = board[i, j] + 1
                    board[i, j+1] = 0
                    points_earned += 2 ** board[i, j]
        return points_earned

    def _add_tile(self):
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

    def display_board(self, axes=None):
        """Displays the board as a matplotlib figure.

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
