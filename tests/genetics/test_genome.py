from genetics.genome import Genome
import unittest


class TestNetworkPlayer(unittest.TestCase):
    def test_similarity(self):
        genome = Genome()
        self.assertEqual(genome.calculate_similarity(genome), 1)

    def test_similarity_and_child(self):
        genome1 = Genome()
        genome2 = Genome()
        child = Genome(mom=genome1, dad=genome2)
        self.assertGreater(child.calculate_similarity(genome1), 0.2)
        self.assertGreater(child.calculate_similarity(genome2), 0.2)


if __name__ == '__main__':
    unittest.main()
