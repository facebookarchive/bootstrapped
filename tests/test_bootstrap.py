# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Tests for bootstrapped'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats


class BootstrappedTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        import warnings
        warnings.filterwarnings('ignore')

    def test_result_math(self):
        mean = 100
        stdev = 10

        samples = np.random.normal(loc=mean, scale=stdev, size=5000)

        bsr = bs.bootstrap(samples, bs_stats.mean)

        self.assertEqual(bsr.value + 1, (bsr + 1).value)
        self.assertEqual(bsr.value + 1, (1 + bsr).value)

        self.assertEqual(bsr.value - 1, (bsr - 1).value)
        self.assertEqual(1 - bsr.value, (1 - bsr).value)

        self.assertEqual(bsr.value * 2, (bsr * 2).value)
        self.assertEqual(2 * bsr.value, (2 * bsr).value)

    def test_bootstrap(self):
        mean = 100
        stdev = 10

        samples = np.random.normal(loc=mean, scale=stdev, size=5000)

        bsr = bs.bootstrap(samples, bs_stats.mean)

        self.assertAlmostEqual(bsr.value, 100, delta=2)
        self.assertAlmostEqual(bsr.upper_bound, 102, delta=2)
        self.assertAlmostEqual(bsr.lower_bound, 98, delta=2)

        bsr2 = bs.bootstrap(samples, bs_stats.mean, alpha=0.1)

        self.assertAlmostEqual(bsr.value, bsr2.value, delta=2)
        self.assertTrue(bsr.upper_bound > bsr2.upper_bound)
        self.assertTrue(bsr.lower_bound < bsr2.lower_bound)

    def test_bootstrap_ab(self):
        mean = 100
        stdev = 10

        test = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl = np.random.normal(loc=mean, scale=stdev, size=5000)
        test = test * 1.1

        bsr = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                              bs_compare.percent_change)
        self.assertAlmostEqual(
            bsr.value,
            10,
            delta=.5
        )

        bsr2 = bs.bootstrap_ab(test, ctrl, bs_stats.sum,
                               bs_compare.percent_change)
        self.assertAlmostEqual(
            bsr2.value,
            -88,
            delta=2
        )

        bsr3 = bs.bootstrap_ab(test, ctrl, bs_stats.sum,
                               bs_compare.percent_change, scale_test_by=10.)
        self.assertAlmostEqual(
            bsr3.value,
            10,
            delta=.5
        )

        test_denom = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl_denom = np.random.normal(loc=mean, scale=stdev, size=5000)
        test_denom = test_denom * 1.1

        bsr4 = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                               bs_compare.percent_change,
                               test_denominator=test_denom,
                               ctrl_denominator=ctrl_denom)
        self.assertAlmostEqual(
            bsr4.value,
            0,
            delta=.5
        )

    def test_compare_functions(self):
        self.assertAlmostEqual(
            bs_compare.percent_change(1.1, 1.),
            10,
            delta=.01
        )
        self.assertAlmostEqual(
            bs_compare.ratio(1.1, 1.),
            1.1,
            delta=0.1
        )
        self.assertAlmostEqual(
            bs_compare.percent_difference(1.1, 1.),
            9.5,
            delta=0.1
        )

        self.assertAlmostEqual(
            bs_compare.difference(1.1, 1.),
            .1,
            delta=0.01
        )

        self.assertAlmostEqual(
            bs_compare.ratio(
                1,
                1.,
            ),
            1.,
            delta=0.1
        )

    def test_bootstrap_ratio(self):
        denom = np.array(([10] * 100) + ([1 / 10.] * 100))
        samples = np.array((([1 / 10.] * 100) + [10] * 100))

        bsr = bs.bootstrap(samples, bs_stats.mean, denominator_values=denom)

        self.assertAlmostEqual(bsr.value, 1, delta=.1)

        bsr = bs.bootstrap(samples / denom, bs_stats.mean)
        self.assertAlmostEqual(bsr.value, 50, delta=5)

    def test_batch_size(self):
        mean = 100
        stdev = 10

        test = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl = np.random.normal(loc=mean, scale=stdev, size=5000)
        test = test * 1.1

        bsr = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                              bs_compare.percent_change)

        bsr_batch = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                                    bs_compare.percent_change,
                                    iteration_batch_size=10)
        self.assertAlmostEqual(
            bsr.value,
            bsr_batch.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_batch.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_batch.upper_bound,
            delta=.1
        )

        bsr = bs.bootstrap(test, bs_stats.mean)

        bsr_batch = bs.bootstrap(test, bs_stats.mean,
                                 iteration_batch_size=10)
        self.assertAlmostEqual(
            bsr.value,
            bsr_batch.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_batch.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_batch.upper_bound,
            delta=.1
        )
