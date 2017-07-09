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
    maxiter : int
        Number of maximum neb calls to find defined tolerance.
    tolerance : float
        Set the maximum error you expect from the model. The lower the more
        exact.
    """
    def __init__(self, initial=None, final=None, tolerance=0.01, maxiter=20):
        self.initialized = False
        self.trained = False
        self.maxiter = maxiter
        self.tolerance = tolerance

        if initial != None and final != None:
            self.initial = read(initial)
            self.final = read(final)
        else:
            print('You need to specify things')

    def initialize(self, calc=None, amp_calc=None, climb=False, intermediates=None, fmax=0.05):
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
        """

        self.calc = calc
        self.amp_calc = amp_calc
        self.intermediates = intermediates
        self.fmax = fmax

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

            self.initial_set = self.run_neb(images, interpolate=True)
            self.initialized = True

        print('NON INTERPOLATED')
        for image in images:
            print(image.get_potential_energy())

        print('INTERPOLATION')
        for image in self.initial_set:
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
            logfile = 'neb_%s.log' % self.iteration
            qn = BFGS(neb, trajectory=self.traj, logfile=logfile)
            qn.run(fmax=self.fmax)
        clean_dir()

    def accelerate(self):
        """This method performs all the acceleration algorithm"""

        nreadimg = -(self.intermediates + 2)
        if self.initialized is True and self.trained is False:
            self.iteration = 0
            self.training_set = self.initial_set
            print('Iteration %s' % self.iteration)
            print('NEB images to slice %s' % nreadimg)
            print('Lenght of training set is %s.' % len(self.training_set))
            label = str(self.iteration)
            amp_calc = self.amp_calc
            amp_calc.set_label(label)
            self.train(self.training_set, amp_calc, label=label)
            del amp_calc
            newcalc = Amp.load('%s.amp' % label)
            images = set_calculators(self.initial_set, newcalc)
            self.run_neb(images)
            print(self.traj)
            ini_neb_images = read(self.traj, index=slice(nreadimg, None))
            del newcalc
            newcalc = Amp.load('%s.amp' % label)
            achieved = self.cross_validate(ini_neb_images, calc=self.calc, amp_calc=newcalc)
            del newcalc

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
            elif self.iteration == self.maxiter:
                break
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
                print('Length of training set is %s.' % len(self.training_set))
            label = str(self.iteration)
            amp_calc = self.amp_calc
            amp_calc.set_label(label)
            self.train(self.training_set, amp_calc, label=label)
            del amp_calc
            newcalc = Amp.load('%s.amp' % label)
            images = set_calculators(self.initial_set, newcalc)
            self.run_neb(images)
            del newcalc
            new_neb_images = read(self.traj, index=slice(nreadimg, None))
            newcalc = Amp.load('%s.amp' % label)
            achieved = self.cross_validate(new_neb_images, calc=self.calc, amp_calc=newcalc)
            del newcalc

    def train(self, trainingset, amp_calc, label=None):
        """This method takes care of training

        Parameters
        ----------
        trainingset : object
            List of images to be trained.
        amp_calc : object
            The Amp instance to do the training of the model.
        label : str
            An integer converted to string.
        """
        if label == None:
            label = str(self.iteration)
        try:
            amp_calc.dblabel = label
            amp_calc.label = label
            amp_calc.train(trainingset)
            #subprocess.call(['mv', 'amp-log.txt', label + '-train.log'])
            del amp_calc
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

def set_calculators(images, calc, label=None):
    """docstring for set_calculators"""

    if label != None:
        print('Label was set to %s' % label)
        calc.label = label
    newimages = []
    for image in images:
        image.set_calculator(calc)
        newimages.append(image)

    return newimages

def clean_dir():
    """Cleaning some directories"""
    print('Cleaning up...')
    remove = [
            'rm',
            '-rf',
            'amp-fingerprint-primes.ampdb',
            'amp-fingerprints.ampdb',
            'amp-log.txt',
            'amp-neighborlists.ampdb'
            ]
    subprocess.call(remove)
    print('Done')
