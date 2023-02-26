import numpy as np
import unittest

def find_eta(segment):
    """
    Calculate eta (pseudorapidity) of a track segment made of hits in a detector
    """
    hit_dim = 3
    segment_hit_num = len(segment)//hit_dim
    delta_rho = segment[(segment_hit_num -1)*hit_dim + 0] - segment[0]
    delta_z = segment[(segment_hit_num - 1)*hit_dim + 2] - segment[2]

    edge_theta = np.arctan2(delta_rho, delta_z)

    edge_eta = - np.log(np.tan(edge_theta/2))
    return edge_eta

class TestEta(unittest.TestCase):
    """
    Test eta calculated properly
    """
    def test_eta(self):
        fake_edge_pos = [1 ,0, 1, 2, 0, 2]
        fake_edge_neg = [1, 0, 1, 2, 0, 0]
        self.assertAlmostEqual(find_eta(fake_edge_pos), 0.88, places = 2)
        self.assertAlmostEqual(find_eta(fake_edge_neg), -0.88, places = 2)
if __name__ == '__main__':
    unittest.main()