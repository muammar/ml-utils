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
    fmax : float
        The maximum force allowed by the optimizer.
    """
    def __init__(self, initial=None, final=None, tolerance=0.01, maxiter=200,
            fmax=0.05, ifmax=0.1, logfile=None):

        if logfile is None:
            logfile = 'acceleration.log'

        self.initialized = False
        self.trained = False
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.fmax = fmax
        self.ifmax = ifmax

        self.logfile = open(logfile, 'w')

        if initial != None and final != None:
            self.initial = read(initial)
            self.final = read(final)
        else:
            self.logfile.write('You need to specify things')

    def initialize(self, calc=None, amp_calc=None, climb=False, intermediates=None):
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
        """

        self.calc = calc
        self.amp_calc = amp_calc
        self.intermediates = intermediates

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

        #print('NON INTERPOLATED')
        #for image in images:
        #    print(image.get_potential_energy())

        #print('INTERPOLATION')
        #for image in self.initial_set:
        #    print(image.get_potential_energy())

    def run_neb(self, images, interpolate=False, fmax=None):
        """This method runs NEB calculation

        Parameters
        ----------
        images : atom objects
            Images created with ASE.
        interpolate : bool
            Interpolate images. Needed when initializing this module.
        """
        if fmax == None:
            fmax = self.ifmax

        self.logfile.write('Fmax is %s \n' % fmax)
        self.logfile.flush()

        neb = NEB(images)

        if interpolate is True:
            neb.interpolate()
            return neb.images
        else:
            self.traj = 'neb_%s.traj' % self.iteration
            logfile = 'neb_%s.log' % self.iteration
            qn = BFGS(neb, trajectory=self.traj, logfile=logfile)
            qn.run(fmax=fmax)
        clean_dir(logfile=self.logfile)

    def accelerate(self):
        """This method performs all the acceleration algorithm"""

        nreadimg = -(self.intermediates + 2)
        if self.initialized is True and self.trained is False:
            self.iteration = 0
            self.training_set = self.initial_set
            self.logfile.write('Iteration %s \n' % self.iteration)
            self.logfile.write('NEB images to slice %s \n' % nreadimg)
            self.logfile.write('Length of training set is %s. \n' % len(self.training_set))
            self.logfile.flush()
            label = str(self.iteration)
            amp_calc = self.amp_calc
            amp_calc.set_label(label)
            self.train(self.training_set, amp_calc, label=label)
            del amp_calc
            newcalc = Amp.load('%s.amp' % label)
            images = set_calculators(self.initial_set, newcalc, logfile=self.logfile)
            self.run_neb(images)
            clean_dir(logfile=self.logfile)
            self.logfile.write('Trajectory file used is %s \n' % self.traj)
            self.logfile.flush()
            ini_neb_images = read(self.traj, index=slice(nreadimg, None))
            del newcalc
            newcalc = Amp.load('%s.amp' % label)
            achieved = self.cross_validate(ini_neb_images, calc=self.calc, amp_calc=newcalc)
            del newcalc
            self.logfile.flush()

        while achieved > self.tolerance:
            self.iteration += 1
            if (self.iteration - 1)  == 0:
                self.logfile.write('INITIAL\n')
                self.logfile.flush()
                ini_neb_images = ini_neb_images[1:-1]
                s = 0
                for _ in ini_neb_images:
                    s += 1
                    self.logfile.write('Adding %s \n' % s)
                    self.training_set.append(_)
                self.logfile.write('Iteration %s \n' % self.iteration)
                self.logfile.write('Length of training set is now %s.\n' % len(self.training_set))
                self.logfile.flush()
            elif self.iteration == self.maxiter:
                break
            else:
                self.traj_to_add = 'neb_%s.traj' % (self.iteration - 1)
                self.logfile.write('Trajectory to be added %s \n' % self.traj_to_add)
                self.logfile.flush()
                images_from_prev_neb = read(self.traj_to_add, index=slice(nreadimg, None))
                images_to_add = images_from_prev_neb[1:-1]
                s = 0
                for _ in images_to_add:
                    s += 1
                    self.logfile.write('Adding %s \n' % s)
                    self.training_set.append(_)
                self.logfile.write('Iteration %s \n' % self.iteration)
                self.logfile.write('Length of training set is %s.\n' % len(self.training_set))
                self.logfile.flush()
            self.training_set = set_calculators(self.training_set, self.calc, logfile=self.logfile)
            label = str(self.iteration)
            amp_calc = self.amp_calc
            amp_calc.set_label(label)
            self.train(self.training_set, amp_calc, label=label)
            del amp_calc
            newcalc = Amp.load('%s.amp' % label)
            images = set_calculators(self.initial_set, newcalc, logfile=self.logfile)
            self.run_neb(images, fmax=self.fmax)
            clean_dir(logfile=self.logfile)
            del newcalc
            new_neb_images = read(self.traj, index=slice(nreadimg, None))
            newcalc = Amp.load('%s.amp' % label)
            achieved = self.cross_validate(new_neb_images, calc=self.calc, amp_calc=newcalc)
            del newcalc
            self.logfile.flush()

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
        self.logfile.write('Length of NEB images %s \n' % len(neb_images))

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
        self.logfile('The metric is %s and tolerance is %s \n '% (metric, self.tolerance))
        self.logfile.flush()
        return metric

def set_calculators(images, calc, label=None, logfile=None):
    """docstring for set_calculators"""

    if label != None:
        self.logfile.write('Label was set to %s\n' % label)
        calc.label = label
    for image in images:
        image.set_calculator(calc)
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

    if logfile is not None:
        logfile.write('Calculator set for %s images\n' % len(images))
        logfile.flush()
    return images

def clean_dir(logfile=None):
    """Cleaning some directories"""
    remove = [
            'rm',
            '-rf',
            'amp-fingerprint-primes.ampdb',
            'amp-fingerprints.ampdb',
            'amp-log.txt',
            'amp-neighborlists.ampdb'
            ]
    subprocess.call(remove)
    if logfile is not None:
        logfile.write('Calculator set for %s images\n' % len(images))
        logfile.flush()
        logfile.write('Cleaning up...\n')
        logfile.flush()
