# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Various comparison statistics functions to run on bootstrap simulations'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as _np


def mean(values, axis=1):
    '''Returns the mean for each row of a matrix'''
    return _np.mean(values, axis=axis)


def sum(values, axis=1):
    '''Returns the sum for each row of a matrix'''
    return _np.sum(values, axis=axis)


def std(values, axis=1):
    '''Returns the standard deviation for each row of a matrix'''
    return _np.std(values, axis=axis)
