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
    similarity = int(fuzz.ratio(str(string1), str(string2)))
    return similarity

def get_pairs(lst):
    """
    Returns a lits of possible pairs in a list.
    """
    pairs = []
    for pair in itertools.product(lst, repeat=2):
        pairs.append(pair)
    pairs = [ list(x) for x in pairs ]
    return pairs

def clustering(lst):
    """
    This method takes a list, then it sorts it and return a new list with
    clustered strings.
    See: http://stackoverflow.com/questions/42184215/clustering-strings-of-a-list-and-return-a-list-of-lists
    """
    clustered = [list(g) for k, g in itertools.groupby(sorted(lst), lambda x: x.split('-')[0])]
    return clustered

def find_minimum(lst):
    """
    Find minimum value inside a list.
    """
    minimum = lst.index(min(lst))
    return minimum
