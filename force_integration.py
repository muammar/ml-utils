#!/usr/bin/env python3
from ase.io import Trajectory
from amp import Amp
import numpy as np

def force_integration(images, amp_calc=None):
    """Compute force integration over images using pure ML

    Parameters
    ----------
    images : str
        Path to images.
    amp_calc : str
        Path to .amp file containing information of the machine-learning model.
    """

    # Loading data
    data = Trajectory(images)

    # Computing references E0 can be 0 or a DFT calcualtion of the reference
    E0 = 0
    p0 = data[0].get_positions()

    # Loading Amp calculator
    amp_calc = Amp.load(amp_calc)


    temp = 0
    for index in range(len(data)):
        image = data[index]
        p1 = image.get_positions()

        if index == 0:
            forcesi = amp_calc.get_forces(data[0])
        else:
            p0 = data[index - 1].get_positions()
            forcesi = amp_calc.get_forces(data[index - 1])

        forcesj = amp_calc.get_forces(image)
        forces = (forcesj + forcesi) / 2

        E = 0.
        for atom in range(len(image)):
            driu = p1[atom] - p0[atom]
            temp += forces[atom].dot(driu)

        E = E0 - temp
        print(E)
        return E


if __name__ == '__main__':

    images = 'test_set2.traj'
    amp_calc = 'kfold.amp'
    force_integration(images, amp_calc=amp_calc)
