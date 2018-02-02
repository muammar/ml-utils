Understanding this example
==========================

This is an example of how the neb acceleration works. In this case, the keyword
arguments `ifmax` (initial fmax) and `step` are there to set a larger `fmax`
than the desired one so that is decreasing as `ifmax / step` making the ML-NEB
to easily converge at the beginning when the training set is poor.

A poor training set could create a situation in which ML-NEB does not converge.
Besides what is described above, one could also set `maxiter` to the number of
maximum iterations that ML-NEB will be let run before finding the desired
`fmax`.
