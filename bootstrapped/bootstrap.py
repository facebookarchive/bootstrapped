# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Functions that allow one to create bootstrapped confidence intervals'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as _np
import warnings as _warnings


class BootstrapResults(object):
    def __init__(self, lower_bound, value, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value = value

    def __str__(self):
        return '{1}    ({0}, {2})'.format(self.lower_bound, self.value,
                                          self.upper_bound)

    def __repr__(self):
        return self.__str__()

    def _apply(self, other, func):
        return BootstrapResults(func(self.lower_bound, other),
                                func(self.value, other),
                                func(self.upper_bound, other))

    def __add__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __radd__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __sub__(self, other):
        return self._apply(float(other), lambda x, other: x - other)

    def __rsub__(self, other):
        return self._apply(float(other), lambda x, other: other - x)

    def __mul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __rmul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)


def _get_alpha_percentiles(alpha):
    return [100 * (alpha / 2.), 50, 100 * (1 - alpha / 2.)]


def _generate_distributions(size, num_iterations):
    # randomly sample value with replacement
    return _np.random.choice(
        size,
        (num_iterations, size),
        replace=True,
    )


def _compute_stat(num, denom, stat_func, sel=None):
    if sel is not None:
        num = num[sel]
        if denom is not None:
            denom = denom[sel]

    result = stat_func(num)
    if denom is not None:
        result /= stat_func(denom)
    return result


def bootstrap(values, stat_func, denominator_values=None, alpha=0.05,
              num_iterations=10000, iteration_batch_size=None,
              return_bootstrap_distribution=False):
    '''Returns bootstrap estimate.
    Args:
        values: numpy array of values to bootstrap
        stat_func: statistic to bootstrap. We provide several default functions:
                * stat_functions.mean
                * stat_functions.sum
                * stat_functions.std
        denominator_values: optional array that does division after the statistic is
            aggregated. This lets you compute group level division statistics.
            One corresponding entry per record in @values.
            Example:
                SUM(value) / SUM(denom) instead of MEAN(value / denom)

                Ex. Cost Per Click
                cost per click across a group
                    SUM(revenue) / SUM(clicks)
                mean cost per click for each
                    MEAN(revenue / clicks)
        alpha: alpha value representing the confidence interval.
            Defaults to 0.05, i.e., 95th-CI.
        num_iterations: number of bootstrap iterations to run. The higher this
            number the more sure you can be about the stability your bootstrap.
            By this - we mean the returned interval should be consistent across
            runs for the same input. This also consumes more memory and makes
            analysis slower.
        iteration_batch_size: The bootstrap sample can generate very large
            matrices. This argument limits the memory footprint by
            batching bootstrap rounds. If unspecified the underlying code
            will produce a matrix of len(values) x num_iterations. If specified
            the code will produce sets of len(values) x iteration_batch_size
            (one at a time) until num_iterations have been simulated.
        return_bootstrap_distribution: Return the bootstrap distribution instead
             of a BootstrapResults object.
    Returns:
        BootstrapResults representing CI and estimated value.
    '''
    if len(values) < 3000:
        _warnings.warn(('This code was designed to work on a large number of '
                        'input samples.\nA heuristic is to have at least 3k '
                        'values.'))

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    results = []

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(iteration_batch_size, num_iterations - rng)

        i = _generate_distributions(len(values), max_rng)

        stat = _compute_stat(values, denominator_values, stat_func, i)

        results.extend(stat)

    if return_bootstrap_distribution:
        return _np.array(results)
    else:
        percentiles = _get_alpha_percentiles(alpha)

        # take the confidence interval
        return BootstrapResults(*_np.percentile(results, percentiles))


def bootstrap_ab(test, ctrl, stat_func, compare_func, test_denominator=None,
                 ctrl_denominator=None, alpha=0.05, num_iterations=10000,
                 iteration_batch_size=None, scale_test_by=1.0,
                 return_bootstrap_distribution=False):
    '''Returns bootstrap confidence intervals for an A/B test.

    Args:
        test: array of test results
        ctrl: array of ctrl results
        stat_func: statistic to bootstrap. We provide several default functions:
                * stat_functions.mean
                * stat_functions.sum
                * stat_functions.std
        compare_func: Function to compare test and control against.
                * compare_functions.difference
                * compare_functions.percent_change
                * compare_functions.ratio
                * compare_functions.percent_difference
        test_denominator: optional array that does division after the statistic is
            aggregated. This lets you compute group level division statistics.
            One corresponding entry per record in test.
            Example:
                SUM(value) / SUM(denom) instead of MEAN(value / denom)

                Ex. Cost Per Click
                cost per click across a group
                    SUM(revenue) / SUM(clicks)
                mean cost per click for each
                    MEAN(revenue / clicks)
        ctrl_denominator: see test_denominator.
        alpha: alpha value representing the confidence interval.
            Defaults to 0.05, i.e., 95th-CI.
        num_iterations: number of bootstrap iterations to run. The higher this
            number the more sure you can be about the stability your bootstrap.
            By this - we mean the returned interval should be consistent across
            runs for the same input. This also consumes more memory and makes
            analysis slower.
        iteration_batch_size: The bootstrap sample can generate very large
            matrices. This function iteration_batch_size limits the memory
            footprint by batching bootstrap rounds.
        scale_test_by: The ratio between test and control population
            sizes. Use this if your test and control split is different from a
            50/50 split. Defaults to 1.0.
        return_bootstrap_distribution: Return the bootstrap distribution instead
             of a BootstrapResults object.

    Returns:
        BootstrapResults representing CI and estimated value.
    '''
    if len(test) < 3000 or len(ctrl) < 3000:
        _warnings.warn(('This code was designed to work on a large number of '
                        'input samples.\nA heuristic is to have at least 3k '
                        'values in both test and ctrl.'))

    if (test_denominator is not None) ^ (ctrl_denominator is not None):
        raise ValueError(('test_denominator and ctrl_denominator must both '
                          'be specified'))

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    results = []

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(rng + iteration_batch_size, num_iterations) - rng

        t_i = _generate_distributions(len(test), max_rng)
        c_i = _generate_distributions(len(ctrl), max_rng)

        test_stat = _compute_stat(test, test_denominator, stat_func, t_i)
        ctrl_stat = _compute_stat(ctrl, ctrl_denominator, stat_func, c_i)

        stat = compare_func(test_stat * scale_test_by, ctrl_stat)

        results.extend(stat)

    if return_bootstrap_distribution:
        return _np.array(results)
    else:
        percentiles = _get_alpha_percentiles(alpha)

        # take the confidence interval
        return BootstrapResults(*_np.percentile(results, percentiles))
