#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import glob

def trained_calculations(path):
    listing = glob.glob(str(path)+'/*.amp')
    listing = [ element for element in listing if not 'initial' in element ]
    failed = [ element for element in listing if 'untrained' in element ]
    trained = set(listing) - set(failed)
    return list(trained)

def common_calculations(list1, list2):
    common = list(set(list1).intersection(list2))
    return list(common)

def listing_calculations(path, extension):
    listing = glob.glob(path+'/*.'+extension)
    return list(listing)
