#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import glob
import shutil
import os
import re

listing = glob.glob(str(sys.argv[1])+'/*.amp')
listing = [ element for element in listing if not 'initial' in element ]
failed = [ element for element in listing if 'untrained' in element ]

print('  Number of Amp successful trainings: ', len(listing))
print('Number of Amp unsuccessful trainings: ', len(failed))

done = 0
remaining = 0

for element in listing:
    fname = re.sub(str(sys.argv[1]), '', element).strip('/')[:-4] + '.traj'
    if os.path.isfile(fname) == True:
        done += 1
    else:
        remaining += 1

print('      Predictions done: ', done)
print('Remaining calculations: ', remaining)
