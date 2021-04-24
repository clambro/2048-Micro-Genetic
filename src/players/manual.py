from game import Action
from players.base import Player
from pynput import keyboard
import time


class ManualPlayer(Player):
    """Play a game of 2048 using manual keyboard inputs."""

    def __init__(self):
        super().__init__()

    def play_game(self, display=True):
        """Play a game of 2048.

        Parameters
        ----------
        display : bool
            Whether to display the graphics. Must be True for manual play.

        Raises
        ------
        ValueError
            If display is False.
        """
        if not display:
            raise ValueError('You need to display the board to play a manual game!')
        super().play_game(True)

    def play_multiple_games(self, num_games, progress_bar=False):
        """Raises an exception since this cannot be done manually.

        Raises
        ------
        TypeError
            Multiple games without graphics cannot be played manually.
        """
        raise TypeError('Multiple games without graphics cannot be played manually.')

    def _choose_action(self, game):
        """Reads an action from keyboard inputs.

        Parameters
        ----------
        game : Game
            The current game state. Unused.

        Returns
        -------
        Action
            The action to take.
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
