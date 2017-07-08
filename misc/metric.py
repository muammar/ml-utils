#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import math

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

def l2(fref, fprediction):
    refdata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fref) ]
    predata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fprediction) ]

    if len(refdata) == len(predata):
        diff = 0.
        for i, _ in enumerate(refdata):
            diff += (abs(float(refdata[i][1]) - float(predata[i][1])))**2

        l2 = math.sqrt(diff)
    else:
        print('The two curves should have the same number of points')
        exit()
    return l2

def rmse(fref, fprediction):
    refdata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fref) ]
    predata = [ line.rstrip('\n').replace(' ',', ').split(', ') for line in open(fprediction) ]

    if len(refdata) == len(predata):
        diff = 0.
        for i, _ in enumerate(refdata):
            diff += (abs(float(refdata[i][1]) - float(predata[i][1])))**2

        rmse = math.sqrt(diff/len(refdata))
    else:
        print('The two curves should have the same number of points')
        exit()
    return rmse

if __name__ == "__main__":

    if '-h' in sys.argv:
        print(message)
        exit()
    else:
        fref = sys.argv[1]
        fprediction = sys.argv[2]
        print(l2(fref, fprediction))
