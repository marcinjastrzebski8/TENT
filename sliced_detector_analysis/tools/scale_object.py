"""
Take any collection of spacial coordinates in the TrackML simulation and normalise them (max abs value = 1).
Coordinates must be in (rho, phi, z). 
"""
import unittest
import numpy as np

#properties of the TrackML dataset hits in barrel detector
max_rho = 1026
max_phi = np.pi
max_z = 1083
max_vals = [max_rho, max_phi, max_z]

def scale_object(object) -> list:
    hit_period = 3
    new_object = []
    for hit in range(int(len(object)/hit_period)):
        for coord in range(hit_period):
            new_object.append(object[hit*hit_period + coord]/max_vals[coord])
    return new_object

class TestScalesCorrectly(unittest.TestCase):
    """
    Test scale_object gives expected values for dummy objects.
    """
    def test_new_df(self):
        halway_edge = [max_rho/2, 0, 0, max_rho, max_phi, max_z]
        full_range_triplet = [0, -max_phi, -max_z, max_rho/2, 0, 0, max_rho, max_phi, max_z]
        scaled_edge = scale_object(halway_edge)
        scaled_triplet = scale_object(full_range_triplet)
        pred_edge = [0.5, 0, 0, 1, 1, 1]
        pred_triplet = [0, -1, -1, 0.5, 0, 0, 1, 1, 1]

        self.assertEqual(scaled_edge, pred_edge)
        self.assertEqual(scaled_triplet, pred_triplet)

if __name__ == '__main__':
    unittest.main()