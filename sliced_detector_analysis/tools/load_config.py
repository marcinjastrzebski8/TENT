import yaml
import unittest
from pathlib import Path


def load_config(file_path = None):
    path = Path(file_path)
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader = yaml.FullLoader)
    #returns the model and the name of the config run
    return cfg, path.stem

#Unit testing
class TestLoad(unittest.TestCase):
    """
    Test config files can be read appropriately on an example config file.
    Note: to test, need to run from top of QuantumKernelEstimation project
    """
    def test_loader(self):
        path_to_example = Path().absolute() / 'QuantumKernelEstimation/sliced_detector_analysis/configs/example.yaml'
        test_variables, filename = load_config(path_to_example)
        print(test_variables)
        self.assertEqual(test_variables['division'], 'new_phi')
        self.assertEqual(filename, 'example')

if __name__ == '__main__':
    unittest.main()