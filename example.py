#!/usr/bin/env python
# -*- coding: utf-8 -*-

from set import (trained_calculations, listing_calculations,
        common_calculations, get_pairs, compare_strings, clustering,
        find_minimum)
from metric import l2
import os

"""
Setting paths to trainings, and predictions done with Amp.
"""
p_training = '/mnt/rdata/data/melkhati/calculations/symmetry_functions/physisorption/corrected/cu221/01_original/train_getpot_g4_cos_NN_1HL/'
p_predictions = '/mnt/rdata/data/melkhati/calculations/symmetry_functions/physisorption/corrected/cu221/01_original/predict/1HL/'
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
#for c  in clusters:
#    if len(c) == 2:
#        print('Some of these failed')
#        print(c)
#    else:
#        print(c)

"""
Convert trajs to data files
"""
#outputf = open('output.txt', 'w')
#min_training = open('mintrain.txt', 'w')

s = 0

with open('output.txt', 'w') as outputf, open('mintrain.txt', 'w') as min_training:
    for c in clusters:
        l2_min = []
        print(c)
        size = len(c)
        for i, _ in enumerate(c):
            call_ase = 'ase-gui -t -g "d(0,4),e" '+ _ +'.traj@0:60:1 > '+ _ +'.data'    # I am only taking 60 images inside this.
            if os.path.isfile(_+'.data') == False:
                os.system(call_ase)

            f = 'File: ' + _ + '\n'
            l2metric = l2(fref, _+'.data')
            outputf.write(f)
            outputf.write(call_ase+'\n')
            outputf.write('L2 = ' + repr(l2metric) + '\n' + '\n' )
            outputf.flush()
            l2_min.append(l2metric)
        minimum = find_minimum(l2_min)
        min_training.write(c[minimum[0]] + ', ' + repr(minimum[1]) + '\n')
        min_training.flush()
        """
        This is for testing purposes
        """
        """
        s += 1
        if s == 1:
            minimum = find_minimum(l2_min)
            min_training.write(c[minimum] + '\n')
            break
        """
outputf.close()
min_training.close()
