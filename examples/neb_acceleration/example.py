import sys
sys.path.append('/path/to/ml-utils/')

from neb import accelerate_neb
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction

from ase.calculators.emt import EMT


initial = 'initial.traj'
final = 'final.traj'

Gs = None
n = 5
cutoff = 6.5
amp_calc = Amp(
        descriptor=Gaussian(cutoff=cutoff, fortran=True, Gs=Gs),
        model=NeuralNetwork(hiddenlayers=(n,n), fortran=True, checkpoints=None)
        )

convergence = {'energy_rmse': 0.0001, 'force_rmse': 0.01}
amp_calc.model.lossfunction = LossFunction(convergence=convergence)

dft_calc = EMT()

neb = accelerate_neb(
        initial=initial,
        final=final,
        tolerance=0.05,
        maxiter=200,
        fmax=0.05,
        ifmax=1.,
        step=2.,
        metric='fmax')

neb.initialize(
        calc=dft_calc,
        amp_calc=amp_calc,
        climb=False,
        intermediates=5
        )
neb.accelerate()
