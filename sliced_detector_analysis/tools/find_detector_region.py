"""
Different ways to divide edges by detector region
NOTE: the squished functionality does not work but is not relevant at the moment#
"""
import numpy as np
from abc import (ABC, abstractmethod)
from . import find_eta

class edge_region_division(ABC):
    def __init__(self,num = 0, squished = False):
        """
        Tuysuz-processed edge data is squished (roughly-normalised).
        When this dataset is used, need to unsquish to match region conditions. 
        """
        self.region_ids = []
        self.number_of_regions = num
        self.squished = squished

    @abstractmethod
    def find_region(self, edge):
        """
        Assign edge to region
        At the moment an edge is judged by the position of it's first hit
        """
        return 
    
    def get_region_ids(self):
        """
        Get list of regions defined for a division
        """
        for i in range(self.number_of_regions):
            self.region_ids.append(str(i))
        return self.region_ids

    def unsquish(self, edge):
        #unit conversion between tuysuz-processed data and trackml for a hit
        conversion_factors = [1000, np.pi, 1000]

        unsquished_edge = []
        for hit in range(2):
            for coord in range(3):
                unsquished_edge.append(edge[hit*3 + coord]*conversion_factors[coord])
        return unsquished_edge

class edge_layer_division(edge_region_division):

    def __init__(self, num = 10, squished = False):
        super().__init__(num, squished)

    def find_region(self, edge):
        if self.squished:
            rho = self.unsquish(edge)[0]
        else:
            rho = edge[0]

        if rho < 50: 
            layer = '0'
        elif (rho >= 50) and (rho < 90):
            layer = '1'
        elif (rho >= 90) and (rho < 150):
            layer = '2'
        elif (rho >= 150) and (rho < 200):
            layer = '3' 
        elif (rho >= 200) and (rho < 300):
            layer = '4'
        elif (rho >= 300) and (rho < 400):
            layer = '5'
        elif (rho >= 400) and (rho < 600):
            layer = '6'
        elif (rho >= 600) and (rho < 750):
            layer = '7'
        elif (rho >= 750) and (rho < 900):
            layer = '8'
        elif (rho >= 900) and (rho < 1050):
            """
            none of the first hits should belong to this layer
            """
            layer = '9'
        else:
            raise ValueError()
        return layer

class edge_phi_division(edge_region_division):
    #THESE WILL HAVE TO BE REMADE SOON
    def __init__(self, num = 10, squished = False):
        super().__init__(num, squished)

    def find_region(self, edge):
        if self.squished:
            phi = self.unsquish(edge)[1]
        else:
            phi = edge[1]

        if (phi >= -np.pi) and (phi < -2.51327412):
            phi_region = '0'
        elif (phi >= -2.51327412) and (phi < -1.88495559):
            phi_region = '1'
        elif (phi >= -1.88495559) and (phi < -1.25663706):
            phi_region = '2'
        elif (phi >= -1.25663706) and (phi < -0.62831853):
            phi_region = '3'
        elif (phi >= -0.62831853) and (phi < 0):
            phi_region = '4'
        elif (phi >= 0) and (phi < 0.62831853):
            phi_region = '5'
        elif (phi >= 0.62831853) and (phi < 1.25663706):
            phi_region = '6'
        elif (phi >= 1.25663706) and (phi < 1.88495559):
            phi_region = '7'
        elif (phi >= 1.88495559) and (phi < 2.51327412):
            phi_region = '8'
        elif (phi >= 2.51327412) and (phi < np.pi):
            phi_region = '9'
        else:
            raise ValueError()
        return phi_region
class edge_new_phi_division(edge_region_division):
    """
    this might become the overlapping division 
    but for now I see no point
    for now just more regions and figured out 
    crossing the discontinuity boundry
    """
    def __init__(self, num = 16, squished = False):
        super().__init__(num, squished)
    def find_region(self, edge):
        if self.squished:
            phi = self.unsquish(edge)[1]
        else:
            phi = edge[1]
        #this range covers 15 out of the 16 regions, last region defined as 
        #whatever doesn't belong here - due to discontinuity in angle definition
        bin_edges = np.linspace(-np.pi + np.pi/self.number_of_regions, np.pi - np.pi/self.number_of_regions, self.number_of_regions)
        #Note: labeling was different with this division until 21/06/22
        #I want '0' to correspond to smallest angles. No real reason but feels cleanest.
        labels = [str(j) for j in range(self.number_of_regions//2 +1, self.number_of_regions)] + [str(i) for i in range(self.number_of_regions//2)] +  [str(self.number_of_regions//2)]

        phi_region = None

        for bin_id in range(len(bin_edges) - 1):
            bin_left_edge =  bin_edges[bin_id]
            bin_right_edge = bin_edges[bin_id+ 1]
            if (phi >= bin_left_edge) and (phi < bin_right_edge):
                phi_region = labels[bin_id]
                break
        if phi_region is None:
            phi_region = labels[-1]
        return phi_region

class edge_z_division(edge_region_division):
    def __init__(self,num = 16, squished = False):
        super().__init__(num, squished)

    def find_region(self, edge):
        if self.squished:
            phi = self.unsquish(edge)[1]
        else:
            phi = edge[1]
        

    def find_region(self, edge):
        if self.squished:
            z = self.unsquish(edge)[2]
        else:
            z = edge[2]

        if (z >= -1100.0) and (z <-880.0):
            z_region = '0'
        elif (z >= -880.0) and (z <-660.0):
            z_region = '1'
        elif (z >= -660.0) and (z <-440.0):
            z_region = '2'
        elif (z >= -440.0) and (z <-220.0):
            z_region = '3'
        elif (z >= -220.0) and (z <-110.0):
            z_region = '4'
        elif (z >= -110.0) and (z <-55.0):
            z_region = '5'
        elif (z >= -55.0) and (z <-27.5):
            z_region = '6'
        elif (z >= -27.5) and (z <0.0):
            z_region = '7'
        elif (z >= 0.0) and (z <27.5):
            z_region = '8'
        elif (z >= 27.5) and (z <55.0):
            z_region = '9'
        elif (z >= 55.0) and (z <110.0):
            z_region = '10'
        elif (z >= 110.0) and (z <220.0):
            z_region = '11'
        elif (z >= 220.0) and (z <440.0):
            z_region = '12'
        elif (z >= 440.0) and (z <660.0):
            z_region = '13'
        elif (z >= 660.0) and (z <880.0):
            z_region = '14'
        elif (z >= 880.0) and (z <1100.0):
            z_region = '15'
        else:
            raise ValueError()
        return z_region
 
class edge_eta_division(edge_region_division):
    def __init__(self, num = 10, squished = False):
        super().__init__(num, squished)
    def find_region(self, edge):
        if self.squished:
            eta = abs(find_eta.find_eta(self.unsquish(edge)))
        else:
            eta = abs(find_eta.find_eta(edge))

        if (eta >= 0) and (eta < 0.2):
            eta_region = '0'
        elif (eta >= 0.2) and (eta < 0.28599383):
            eta_region = '1'
        elif (eta >= 0.28599383) and (eta < 0.40896235):
            eta_region = '2'
        elif (eta >= 0.40896235) and (eta < 0.58480355):
            eta_region = '3'
        elif (eta >= 0.58480355) and (eta < 0.83625103):
            eta_region = '4'
        elif (eta >= 0.83625103) and (eta < 1.19581317):
            eta_region = '5'
        elif (eta >= 1.19581317) and (eta < 1.70997595):
            eta_region = '6'
        elif (eta >= 1.70997595) and (eta < 2.44521285):
            eta_region = '7'
        elif (eta >= 2.44521285) and (eta < 3.49657893):
            eta_region = '8'
        elif (eta >= 3.49657893): #I think some rare edges will appear here 
            eta_region = '9'
        else:
            raise ValueError()
        return eta_region

"""
Consider these unit tests.

fake_edge = [0.1,-0.1,-0.2, 0.1, -0.2, -0.2]

layer_div = edge_layer_division(squished = True)
z_div = edge_z_division(squished = True)
phi_div = edge_phi_division(squished = True)

edge_layer = layer_div.find_region(fake_edge)
edge_z = z_div.find_region(fake_edge)
edge_phi = phi_div.find_region(fake_edge)

print(edge_layer)
print(edge_z)
print(edge_phi)

print(layer_div.get_region_ids())
print(z_div.get_region_ids())
"""