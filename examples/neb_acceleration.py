import sys
sys.path.append('/home/muammar/github/amp-utils/')

from neb import accelerate_neb
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction

from ase.calculators.emt import EMT


initial = '../images/initial.traj'
final = '../images/final.traj'

Gs = None
n = 5
cutoff = 6.5
amp_calc = Amp(
        descriptor=Gaussian(cutoff=cutoff, fortran=True, Gs=Gs),
        model=NeuralNetwork(hiddenlayers=(n,n), fortran=True)
        )

convergence = {'energy_rmse': 0.02, 'force_rmse': 0.04}
amp_calc.model.lossfunction = LossFunction(convergence=convergence)

dft_calc = EMT()

neb = accelerate_neb(
        initial=initial,
        final=final,
        tolerance=0.021,
        maxiter=200,
        fmax=0.05,
        ifmax=1.35,
        step=3.,
        maxrunsteps=50,
        metric='fmax'
        )

neb.initialize(
        calc=dft_calc,
        amp_calc=amp_calc,
        climb=False,
        intermediates=5,
        )

neb.accelerate()
