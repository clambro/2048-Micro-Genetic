from genetics.genome import Genome
from players.base import Player


class NetworkPlayer(Player):
    """A feed-forward neural network to play 2048.

    Attributes
    ----------
    generation : int
        Which generation the network belongs to.
    genome : Genome
        The genome containing the weights for the network, along with rules for reproduction.
    """

    def __init__(self, gen=0, mom=None, dad=None, genome=None):
        """Builds the network from a genome if given, or two parents, falling back to random generation if neither.

        Parameters
        ----------
        gen : int
            The current generation.
        mom : Optional[Net]
            A net from which the chromosome will be sampled.
        dad : Optional[Net]
            The other net from which the chromosome will be sampled.
        genome : Optional[ndarray]
            The genome containing the network weights.
        """
        super().__init__()
        self.generation = gen
        if genome is not None:
            self.genome = genome
        elif None not in [mom, dad]:
            self.genome = Genome(mom.genome, dad.genome)
        else:
            self.genome = Genome()

    def _choose_action(self, game):
        """Evaluate the position using the network and choose the best legal move it determines.

        Parameters
        ----------
        game : Game
            The current game state.

        Returns
        -------
        best_move : Action
            The action to take.
        """
        legal_moves = game.get_legal_moves()
        sorted_moves = self.genome.calculate_move_order(game.board)
        for move in sorted_moves:
            if move in legal_moves:
                return move
