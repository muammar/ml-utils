#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import glob
from fuzzywuzzy import fuzz
import itertools

def trained_calculations(path):
    """
    This function returns a list of successfully trained Amp calculations.
    """
    listing = glob.glob(str(path)+'/*.amp')
    listing = [ element for element in listing if not 'initial' in element ]
    failed = [ element for element in listing if 'untrained' in element ]
    trained = set(listing) - set(failed)
    return list(trained)

def common_calculations(list1, list2):
    """
    Return common set of elements in lists.
    """
    common = list(set(list1).intersection(list2))
    return list(common)

def listing_calculations(path, extension):
    """
    List files inside path with extension=extension.
    """
    listing = glob.glob(path+'/*.'+extension)
    return list(listing)

def compare_strings(string1, string2):
    """
    Compare similarity between strings. Useful for doing some clustering work.
    """
    similarity = fuzz.ratio(str(string1), str(string2))
    return similarity

def get_pairs(listofelements):
    """
    Returns a lits of possible pairs in a list.
    """
    pairs = []
    for pair in itertools.product(listofelements, repeat=2):
        pairs.append(pair)
    pairs = [list(x) for x in pairs]
    return pairs
