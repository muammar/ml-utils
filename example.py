#!/usr/bin/env python
# -*- coding: utf-8 -*-

from set import (trained_calculations, listing_calculations,
        common_calculations, get_pairs, compare_strings, clustering)
from metric import l2
import os

"""
Setting paths to trainings, and predictions done with Amp.
"""
p_training = '/mnt/rdata/data/melkhati/calculations/symmetry_functions/physisorption/corrected/cu221/01_original/train_getpot_g4_cos_NN_2HL/'
p_predictions = '/mnt/rdata/data/melkhati/calculations/symmetry_functions/physisorption/corrected/cu221/01_original/predict/2HL/'
fref = '/mnt/rdata/data/melkhati/calculations/symmetry_functions/physisorption/corrected/cu221/01_original/get_potential/potential_energy.dat'

trained = trained_calculations(p_training)
predict = listing_calculations(p_predictions, 'traj')

trained = [ element[126:-4] for element in trained if '.amp' in element ]
predict = [ element[111:-5] for element in predict if '.traj' in element ]

"""
Create common list of calculations
"""
commons = common_calculations(trained, predict)

"""
Clustering
"""
clusters = clustering(commons)
for c  in clusters:
    if len(c) == 2:
        print(c)

"""
Convert trajs to data files
"""
s = 0
for c in clusters:
    l2_min = []
    for _ in c:
        call_ase = 'ase-gui -t -g "d(0,4),e" '+ _ +'.traj@0:60:1 > '+ _ +'.data'    # I am only taking 60 images inside this.
        if os.path.isfile(_+'.data'):
            pass
        else:
            os.system(call_ase)

        print('File:', _)
        print(call_ase)
        print('L2 =',l2(fref, _+'.data'))
        l2_min.append(l2(fref, _+'.data'))
        print()
    s += 1
    if s == 1:
        print(l2_min)
        break

