#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.io import Trajectory
import random


def split(images, trainingname='trainingimages.traj',
          testname='testimages.traj', shuffle=True, test_set=20):
    """Split data set in training and test sets.

    images : str
        Path to images to be split.
    trainingname : str
        Name of the training set trajectory file. By default is
        trainingimages.traj
    testname : str
        Name of the test set trajectory file. By default is
        testimages.traj
    test_set : integer
        Porcentage of training data that will be used as test set.
    shuffle : bool
        Whether or not the data will be randomized.
    """
    images = Trajectory(images)

    total_length = len(images)
    test_length = (test_set * total_length / 100)
    training_leght = total_length - test_length

    _images = range(len(images))

    if shuffle is True:
        random.shuffle(_images)

    trainingimages = []
    ti = Trajectory(trainingname, mode='w')

    log = open('log.txt', 'w')

    for i in _images[0:training_leght]:
        trainingimages.append(i)
        ti.write(images[i])
    log.write(str(trainingimages))
    log.write('\n')

    testimages =[]
    test = Trajectory(testname, mode='w')
    for i in _images[-test_length:-1]:
        testimages.append(i)
        test.write(images[i])

    log.write(str(testimages))
    log.close()
    return
