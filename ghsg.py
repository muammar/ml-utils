from ase.io import Trajectory
from amp.utilities import hash_images
from amp import Amp
import numpy as np
from scipy.special import erf
from collections import OrderedDict


class GHSG(object):
    """This class is based on the `Interatomic potentials for ionic systems
    with density functional accuracy based on charge densities obtained by
    a neural network` by Ghasemi et al. DOI: 10.1103/PhysRevB.92.045131

    Parameters
    ----------
    images : str
        Path to ASE trajectory file.
    descriptor : object
        Descriptor object created by Amp.
    calc : str
        Path to amp calculator.
    charge : float
        Charge constrain to perform the Lagrange minimization.
    """
    def __init__(self, images, descriptor, calc=None, charge=0):
        _images = Trajectory(images)
        self.images = hash_images(_images)
        self.descriptor = descriptor
        self.calc = calc
        self.charge = charge

    def calculate(self):
        """docstring for calculate"""

        Gamma = {'Na': 1.0, 'Cl': 2.0}
        Jii = {'Na': 0.2, 'Cl': 0.1}
        Aij_matrix = []

        hashes = self.images.keys()
        hashes = ['6b1ec1314b566b7fd3a01e762bf19f16']
        for hash in hashes:
            print(hash)
            EN_dict, EN_vector = self.get_atomic_electronegativities(hash,
                                                                     self.calc)

            image = self.images[hash]
            E = image.get_potential_energy()
            print(E)
            Ei = {'Na': -3.53687931764, 'Cl': -2.88586245501}

            for i, atomi in enumerate(image):
                for j, atomj in enumerate(image):
                    rij = image.get_distance(i, j)
                    a = self.Aij(i, j, atomi, atomj, Gamma=Gamma, Jii=Jii,
                                 rij=rij)
                    Aij_matrix.append(a)

            size = len(image)

            Aij_matrix = np.array(Aij_matrix).reshape(size, size)

            # Now we create the required block matrices

            Aij_matrix = np.block([[Aij_matrix, np.ones((size, 1))],
                                   [np.ones((1, size + 1))]
                                   ])

            Aij_matrix[-1][-1] = 0.
            EN_vector = np.append(EN_vector, self.charge)
            Q = np.linalg.solve(Aij_matrix, EN_vector)
            print('Total Charge: {}'.format(sum(Q[0:-1])))
            print('Charge per atom')
            print(EN_dict.keys())
            print(Q[0:-1])

            u1 = 0.
            for i, atom in enumerate(image):
                symbol = atom.symbol
                ei = Ei[symbol]
                xi = EN_dict[(i, symbol)]
                qi = Q[i]
                jii = Jii[symbol]
                gii = Gamma[symbol]
                u1 += ei + ((xi * qi) + (.5 * (jii + ((2 * gii) / np.sqrt(np.pi))) * qi ** 2))

            u2 = 0.
            for i, atomi in enumerate(image):
                for j, atomj in enumerate(image):
                    if i > j:
                        qi = Q[i]
                        qj = Q[j]
                        rij = image.get_distance(i, j)
                        gamma = self.get_gamma(atomi, atomj, Gamma)
                        u2 += (qi * qj) * ((erf(gamma * rij) / rij))

            u = u1 + u2

            print('Total Energy GHSG: {}' .format(u))
            #print('Total Energy DFT:  {}' .format(image.get_potential_energy()))
            return u

    def Aij(self, i, j, atomi, atomj, Gamma=None, rij=None, Jii=None):
        """
        Parameters
        ----------
        i : int
            Index of central atom.
        j : int
            Index of neighbor atom.
        atomi : object
            Atom object of central atom.
        atomj : object
            Atom object of neighbor atom.
        Gamma : dict
            Dictionary with gaussian width per atom.
        rij : float
            Distance between atom i and j.
        Jii : dict
            Dictionary with hardness per atom.

        Returns
        -------
        aij : float
            Matrix element aij.

        """
        gamma = self.get_gamma(atomi, atomj, Gamma)

        if i == j:
            aij = Jii[atomi.symbol] + (2 * gamma / np.sqrt(np.pi))
        else:
            aij = erf(gamma * rij) / rij
        return aij

    def get_gamma(self, atomi, atomj, Gamma):
        """
        Parameters
        ----------
        atomi : object
            Atom object of central atom.
        atomj : object
            Atom object of neighbor atom.
        Gamma : dict
            Dictionary with gaussian width per atom.

        Returns
        -------
        gamma : float
            Value of gammaij
        """
        gammai = np.square(Gamma[atomi.symbol])
        gammaj = np.square(Gamma[atomj.symbol])

        gamma = 1 / np.sqrt(gammai + gammaj)

        return gamma

    def get_atomic_electronegativities(self, hash, calc):
        """
        Returns
        -------
        atomic_electronegativity : dict
            Dictionary with keys (index, symbol) and value  (atomic
            electronegativity).
        electronegativity_vector : list
            List of electronegativities.
        """
        atomic_electronegativity = OrderedDict()
        electronegativity_vector = []

        # calculating fingerprints
        self.descriptor.calculate_fingerprints(self.images)
        self.descriptor.fingerprints.open()

        fingerprints = self.descriptor.fingerprints[hash]

        # Load Amp calculator
        nn_calc = Amp.load(self.calc)

        for index, (symbol, afp) in enumerate(fingerprints):
            en = nn_calc.model.calculate_atomic_energy(afp, index, symbol)
            atomic_electronegativity[(index, symbol)] = en
            electronegativity_vector.append(en)

        electronegativity_vector = np.array(electronegativity_vector)

        return atomic_electronegativity, electronegativity_vector

if __name__ == '__main__':
    from amp.descriptor.gaussian import Gaussian
    descriptor = Gaussian(Gs=None)
    images = '/home/muammar/Dropbox/NUC/electro/_ok/nacl_en.traj'
    calc = '/home/muammar/Dropbox/NUC/electro/train/nn.amp'
    GHSG = GHSG(images, descriptor, calc, charge=0)
    for charge in [0]:
        GHSG.charge=charge
        GHSG.calculate()
