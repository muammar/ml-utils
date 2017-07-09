# This module was written by Muammar El Khatib <muammar@brown.edu>

# ASE imports
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS

# Amp imports
from amp import Amp

from sklearn.metrics import mean_absolute_error
import subprocess

class accelerate_neb(object):
    """Accelerating NEB calculations using Machine Learning

    This class accelerates NEB calculations using an algorithm proposed by
    Peterson, A. A.  (2016). Acceleration of saddle-point searches with machine
    learning. The Journal of Chemical Physics, 145(7), 74106.
    https://doi.org/10.1063/1.4960708

    Parameters
    ----------
    initial : str
        Path to the initial image.
    final : str
        Path to the final image.
    """
    def __init__(self, initial=None, final=None):
        self.initialized = False
        self.trained = False

        if initial != None and final != None:
            self.initial = read(initial)
            self.final = read(final)
        else:
            print('You need to specify things')

    def initialize(self, calc=None, amp_calc=None, climb=False, intermediates=None, fmax=0.05, tolerance=0.01):
        """Method to initialize the acceleration of NEB

        Parameters
        ----------
        calc : object
            This is the calculator used to perform DFT calculations.
        amp_calc : object
            This is the machine learning model used to perform predictions.
        intermediates : int
            Number of intermediate images for the NEB calculation.
        climb : bool
            Whether or not NEB will be run using climbing image mode.
        fmax : float
            The maximum force allowed by the optimizer.
        tolerance : float
            What is the maximum error you expect from the model. The lower the more
            exact.
        """

        self.calc = calc
        self.amp_calc = amp_calc
        self.intermediates =  intermediates
        self.fmax = fmax
        self.tolerance = tolerance

        images = [ self.initial ]

        if self.initialized is False:
            for intermediate in range(self.intermediates):
                image = self.initial.copy()
                image.set_calculator(self.calc)
                image.get_potential_energy()
                image.get_forces()
                #image.set_constraint(self.calc)
                images.append(image)

            images.append(self.final)

            self.training_set = self.run_neb(images, interpolate=True)
            self.initialized = True

        print('NON INTERPOLATED')
        for image in images:
            print(image.get_potential_energy())

        print('INTERPOLATION')
        for image in self.training_set:
            print(image.get_potential_energy())

    def run_neb(self, images, interpolate=False):
        """This method runs NEB calculation

        Parameters
        ----------
        images : atom objects
            Images created with ASE.
        interpolate : bool
            Interpolate images. Needed when initializing this module.
        """
        neb = NEB(images)

        if interpolate is True:
            neb.interpolate()
            return neb.images
        else:
            self.traj = 'neb_%s.traj' % self.iteration
            qn = BFGS(neb, trajectory=self.traj)
            qn.run(fmax=self.fmax)

    def accelerate(self):
        """This method performs all the acceleration algorithm"""

        nreadimg = -(self.intermediates + 2)
        if self.initialized is True and self.trained is False:
            self.iteration = 0
            print('NEB images to slice %s' % nreadimg)
            print('Lenght of training set is %s.' % len(self.training_set))
            print('Iteration %s' % self.iteration)
            self.train(self.training_set, self.amp_calc, iteration=str(self.iteration))
            calc = Amp.load(str(self.iteration) + '.amp')
            images = set_calculators(self.training_set, calc)
            self.run_neb(images)
            print(self.traj)
            ini_neb_images = read(self.traj, index=slice(nreadimg, None))
            achieved = self.cross_validate(ini_neb_images, calc=self.calc, amp_calc=self.amp_calc)

        while achieved > self.tolerance:
            self.iteration += 1
            if (self.iteration - 1)  == 0:
                print('INITIAL')
                ini_neb_images = ini_neb_images[1:-1]
                s = 0
                for _ in ini_neb_images:
                    s += 1
                    print('adding %s' % s)
                    self.training_set.append(_)
                print('Iteration %s' % self.iteration)
                print('Lenght of training set is %s.' % len(self.training_set))
            else:
                print('ITER > 1')
                self.traj_to_add = 'neb_%s.traj' % (self.iteration - 1)
                print(self.traj_to_add)
                images_to_add = read(self.traj_to_add, index=slice(nreadimg, None))
                images_to_add = images_to_add[1:-1]
                s = 0
                for _ in images_to_add:
                    s += 1
                    print('adding %s' % s)
                    self.training_set.append(_)
                print('Iteration %s' % self.iteration)
                print('Lenght of training set is %s.' % len(self.training_set))
            self.train(self.training_set, self.amp_calc, iteration=str(self.iteration))
            calc = Amp.load(str(self.iteration) + '.amp')
            images = set_calculators(self.training_set, calc)
            self.run_neb(images)
            ini_neb_images = read(self.traj, index=slice(nreadimg, None))
            achieved = self.cross_validate(ini_neb_images, calc=self.calc, amp_calc=self.amp_calc)

    def train(self, trainingset, amp_calc, iteration=None):
        """This method takes care of training """
        try:
            amp_calc.dblabel = iteration
            amp_calc.label = iteration
            amp_calc.train(trainingset)
            subprocess.call(['mv', 'amp-log.txt', iteration + '-log.txt'])
        except:
            raise

    def cross_validate(self, neb_images, calc=None, amp_calc=None):
        """Cross validate

        This method will verify whether or not a metric to measure error
        between predictions and targets meets the desired criterium. It uses
        metrics found in scikit-learn module.

        Parameters
        ----------
        neb_images : objects
            These are the images read after a successful NEB calculation.
        calc : object
            This is the calculator used to perform DFT calculations.
        amp_calc : object
            This is the machine learning model used to perform predictions.
        """
        print(len(neb_images))

        amp_energies = []
        amp_images = set_calculators(neb_images, amp_calc)

        for image in amp_images:
            energy = image.get_potential_energy()
            amp_energies.append(energy)

        dft_energies = []
        dft_images = set_calculators(neb_images, calc)
        for image in dft_images:
            energy = image.get_potential_energy()
            dft_energies.append(energy)

        metric = mean_absolute_error(dft_energies, amp_energies)
        print(metric)
        return metric

def set_calculators(images, calc):
    """docstring for set_calculators"""
    newimages = []

    for image in images:
        image.set_calculator(calc)
        newimages.append(image)

    return newimages
