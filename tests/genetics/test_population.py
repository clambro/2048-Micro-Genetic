from copy import copy
from genetics.population import Population
from tempfile import NamedTemporaryFile
import unittest


class TestPopulation(unittest.TestCase):
    def test_init_no_params(self):
        with self.assertRaises(ValueError) as e:
            Population()
        self.assertIn('If pop is none, then both num_nets', str(e.exception))

    def test_init_new_pop(self):
        p = Population(num_nets=3, num_elite=1)
        self.assertEqual(len(p.networks), 3)
        self.assertLess(p.similarity, 0.7)

    def test_init_from_pop(self):
        p = Population(num_nets=3, num_elite=1)
        p = Population(num_nets=10, num_elite=3, pop=p)  # First two parameters should be ignored.
        self.assertEqual(len(p.networks), 2)
        self.assertEqual(len(p.elites), 1)
        self.assertEqual(p.generation, 2)
        self.assertGreater(p.similarity, 0.5)

    def test_randomize(self):
        p1 = Population(num_nets=3, num_elite=1)
        p1.elites = [p1.networks.pop()]
        p2 = copy(p1)
        p2.randomize()
        self.assertEqual(p1.elites, p2.elites)
        self.assertNotEqual(p1.networks, p2.networks)

    def test_play_games(self):
        p = Population(num_nets=3, num_elite=1)
        p.elites = [p.networks.pop()]

        p.play_games(2, include_elites=True, progress_bar=False)
        self.assertEqual(p.elites[0].get_num_games_played(), 2)
        [self.assertEqual(n.get_num_games_played(), 2) for n in p.networks]

        p.play_games(2, include_elites=True, progress_bar=False, thresh=1e20)  # Impossibly high threshold.
        self.assertEqual(p.elites[0].get_num_games_played(), 2)
        [self.assertEqual(n.get_num_games_played(), 2) for n in p.networks]

        p.play_games(2, include_elites=False, progress_bar=False)
        self.assertEqual(p.elites[0].get_num_games_played(), 2)
        [self.assertEqual(n.get_num_games_played(), 4) for n in p.networks]

    def test_get_sorted_networks(self):
        p = Population(num_nets=3, num_elite=1)
        p.elites = [p.networks.pop()]
        p.play_games(2, include_elites=True, progress_bar=False)
        self.assertEqual(len(p.get_sorted_networks(False)), 2)
        self.assertEqual(len(p.get_sorted_networks(True)), 3)

    def test_save(self):
        p = Population(num_nets=3, num_elite=1)
        with NamedTemporaryFile() as f:
            p.save(f.name)


if __name__ == '__main__':
    unittest.main()
