#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

message = """
This script computes the absolute value of the difference between the ab initio
computed energies (one by one) with respect to the predicted ones. Then it
performs the sum of all those differences and takes the square root of it.


The nearer to zero the sum of these differences is, the more similar the curves
are.

To use it:

    python l2_metric.py /path/to/dft.traj /path/to/amp_prediction.traj


See: https://proofwiki.org/wiki/Definition:L2_Metric
"""

from ase.io import read
import sys
import math

if '-h' in sys.argv:
    print(message)
    exit()

fref = sys.argv[1]
fprediction = sys.argv[2]


refdata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fref) ]
preddata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fprediction) ]


diff = 0.
for i, _ in enumerate(refdata):
    diff += (abs(float(refdata[i][1]) - float(preddata[i][1])))**2

l2 = math.sqrt(diff)
print(l2)
