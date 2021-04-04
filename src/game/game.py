from game.action import Action
from game.board import Board
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
    """

    def __init__(self):
        """Sets up the board and clears the key-state."""
        self.board = Board()
        self.score = 0
        self.highest_tile = 0
        self.moves = 0
        self.game_over = False

    def perform_action(self, direction):
        """Move in the given direction and update the game parameters. Ignore illegal moves.

        Parameters
        ----------
        direction : Action
            The action to take in the game state.
        """
        move_was_legal, points_earned = self.board.move(direction)
        if move_was_legal:
            self.moves += 1
            self.score += points_earned
            self.highest_tile = 2 ** np.max(self.board.board)
        return move_was_legal

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
                time.sleep(0.1)  # Add a small delay between reads to avoid multiple moves per key press.
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
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)
            plt.figtext(0, 0.01, 'Use the arrow keys to move or pres ESC to quit.')
            show_flag = True
            plt.pause(1)  # Gives you a bit of time to adjust the plot
        else:
            show_flag = False
            axes.clear()
        axes.set_title('Score = ' + str(self.score))
        plt.imshow(self.board.board, cmap='summer')
        for (j, i), label in np.ndenumerate(2**self.board.board):
            if label != 1:
                axes.text(i, j, label, ha='center', va='center')
        plt.pause(0.001)  # Need a slight delay or the plot won't render
        if show_flag:
            plt.show()
        return axes

    def play_game(self):
        """Play a game of 2048."""
        ax = self.display_game()
        while not self.board.game_over:
            key = self.read_key()
            if key == Action.QUIT:
                break
            self.perform_action(key)
            self.display_game(ax)
        plt.close()
        print(f'Game Over. Final score was {self.score}. Highest tile was {self.highest_tile}.')
