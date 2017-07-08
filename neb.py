# This module was written by Muammar El Khatib <muammar@brown.edu>
from ase.io import read
from ase.neb import NEB
from ase.optimize import QuasiNewton

from sklearn.metrics import mean_absolute_error


class accelerate_neb(object):
    """Accelerating NEB calculations using Machine Learning

    This module accelerates NEB calculations using an algorithm proposed by
    Peterson, A. A.  (2016). Acceleration of saddle-point searches with machine
    learning. The Journal of Chemical Physics, 145(7), 74106.
    https://doi.org/10.1063/1.4960708

    Parameters
    ----------
    initial : str
        Path to the initial image.
    final : str
        Path to the final image.
    intermediates : int
        Number of intermediate images for the NEB calculation.
    climb : bool
        Whether or not NEB will be run using climbing image mode.
    """
    def __init__(self, initial=None, final=None, climb=False,
            intermediates=None):
        self.initialized = False
        self.trained = False
        self.intermediates =  intermediates

        if initial != None and final != None:
            self.initial = read(initial)
            self.final = read(final)
        else:
            print('You need to specify things')

    def initialize(self, calc=None, amp_calc=None):
        """Method to initialize the acceleration of NEB

        Parameters
        ----------
        calc : object
            This is the calculator used to perform DFT calculations.
        amp_calc : object
            This is the machine learning model used to perform predictions.
        """

        self.calc = calc
        self.amp_calculator = amp_calc
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
            qn = BFGS(neb, trajectory='neb.traj')
            qn.run(fmax=0.05)

    def accelerate(self):
        """This method performs all the acceleration algorithm"""

        if self.initialized is True and self.trained is False:
            self.train(self.initial_set, self.amp_calculator)

    def train(self, trainingset, amp_calculator, iteration=None):
        """This method takes care of training """
        try:
            amp_calculator.train(trainingset)
        except:
            raise

    def cross_validate(self, amp_predictions, dft_calculations):
        """Cross validate

        This method will verify whether or not a metric to measure error
        between predictions and targets meets the desired criterium. It uses
        metrics found in scikit-learn module.
        """
        metric = mean_absolute_error(dft_calculations, amp_predictions)
        return metric
