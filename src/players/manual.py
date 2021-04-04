from game.action import Action
from game.game import Game
import matplotlib.pyplot as plt
from players.base import Player
from pynput import keyboard
import time


class ManualPlayer(Player):
    """

    """

    def __init__(self):
        """"""
        super().__init__()

    def play_game(self, display=True):
        """"""
        super().play_game(True)

    @staticmethod
    def choose_action(game):
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
