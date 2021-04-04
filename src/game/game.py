from action import Action, DIRECTIONS
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard
import time


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
    moves : int
        The number of moves made in the current game.
    game_over : bool
        Whether or not the game is over.
    last_move_illegal : bool
        Whether or not the last attempted move was illegal.
    """

    def __init__(self):
        """Sets up the board and clears the key-state."""
        self.board = np.zeros((4, 4), dtype=np.int)
        self.add_tile()
        self.add_tile()
        self.score = 0
        self.highest_tile = 0
        self.moves = 0
        self.game_over = False
        self.last_move_illegal = False

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
        if self.board.all() and self.no_moves():
            self.game_over = True

    def no_moves(self):
        """Return True if there are no legal moves, else False."""
        orig_board = np.copy(self.board)
        orig_score = np.copy(self.score).tolist()
        for direction in DIRECTIONS:
            self.move(direction)
            self.board = orig_board
            self.score = orig_score
            if not self.last_move_illegal:
                return False
            else:
                self.last_move_illegal = False
        return True

    @staticmethod
    def read_key():
        """Reads a key from keyboard inputs.

        Returns
        -------
        key : Action
            The key pressed, represented as an Action.
        """
        with keyboard.Events() as events:
            for event in events:
                time.sleep(0.1)  # Add a small delay between reads to avoid multiple moves per key.
                if event.key == keyboard.Key.left:
                    return Action.LEFT
                elif event.key == keyboard.Key.right:
                    return Action.RIGHT
                elif event.key == keyboard.Key.up:
                    return Action.UP
                elif event.key == keyboard.Key.down:
                    return Action.DOWN
                elif event.key == keyboard.Key.esc:
                    return Action.QUIT

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

    def move(self, key):
        """Execute a move in the direction of key.

        Rotate the board so that key points leftwards, move left, then rotate back to the original position.

        Parameters
        ----------
        key : int
            The ordinal value of the key pressed.
        """
        prev_board = np.copy(self.board)
        if key == Action.LEFT:
            rot = 0
        elif key == Action.UP:
            rot = 1
        elif key == Action.RIGHT:
            rot = 2
        else:
            rot = -1
        self.board = np.rot90(self.board, rot)
        # Slide, merge, slide pattern reflects the official 2048 rules
        self.slide_left()
        self.merge_left()
        self.slide_left()
        self.board = np.rot90(self.board, -1*rot)
        # If the board is unchanged by the move, then it was illegal.
        if np.array_equal(self.board, prev_board):
            self.last_move_illegal = True

    def process_key(self, key):
        """Quit, or move according to key. Ignore illegal moves.

        Parameters
        ----------
        key : int
            The ordinal value of the key pressed.
        """
        if key == Action.QUIT:
            self.game_over = True
        else:
            self.move(key)
            if not self.last_move_illegal:
                self.add_tile()
                self.moves += 1
        self.highest_tile = np.max(2**self.board)

    def display_game(self, axes=None):
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

    def play_game(self):
        """Play a game of 2048."""
        ax = self.display_game()
        while not self.game_over:
            key = self.read_key()
            self.last_move_illegal = False
            self.process_key(key)
            self.display_game(ax)
        plt.close()
        print('Game Over')
