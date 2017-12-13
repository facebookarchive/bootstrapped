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
import bootstrapped.permutation as pm
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import scipy.sparse as sparse
import scipy.stats as st


def to_sparse(data):
    mat = sparse.csr_matrix((1, len(data)))
    mat[0, :len(data)] = data
    return mat

class BootstrappedTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        import warnings
        warnings.filterwarnings('ignore')

    def test_bootstrap_results(self):
        bsr = bs.BootstrapResults(-1, 0, 1)
        self.assertEqual(bsr.error_width(), 2)
        self.assertEqual(bsr.error_fraction(), np.inf)
        self.assertEqual((bsr + 1).error_fraction(), 2)

        self.assertEqual(bsr.is_significant(), False)
        self.assertEqual((bsr + 2).is_significant(), True)
        self.assertEqual((bsr - 2).is_significant(), True)
        self.assertEqual(bsr.get_result(), 0)
        self.assertEqual((bsr + 2).get_result(), 1)
        self.assertEqual((bsr - 2).get_result(), -1)

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

    def test_randomized_permutation(self):
        ctrl = np.array([54, 51, 58, 44, 55, 52, 42, 47, 58, 46])
        test = np.array([54, 73, 53, 70, 73, 68, 52, 65, 65])

        pmr = pm.permutation_test(test, ctrl, bs_stats.mean,
                                   bs_compare.difference, num_iterations=10000)
        self.assertAlmostEqual(
            pmr,
            0.001,
            delta=0.0015
        )

        ctrl = np.array([56, 348, 162, 420, 440, 250, 389, 476, 288, 456])
        test = np.array([69, 361, 175, 433, 453, 263, 402, 489, 301, 469])

        pmr2 = pm.permutation_test(test, ctrl, bs_stats.mean,
                                   bs_compare.difference, num_iterations=10000)
        self.assertAlmostEqual(
            pmr2,
            0.83,
            delta=0.005
        )

        test = np.array([37, 49, 55, 57])
        ctrl = np.array([23, 31, 46])

        pmr3 = pm.permutation_test(test, ctrl, bs_stats.mean,
                                   bs_compare.difference, num_iterations=10000)
        self.assertAlmostEqual(
            pmr3,
            0.114,
            delta=0.01
        )

        test = np.array([12.6, 11.4, 13.2, 11.2, 9.4, 12.0])
        ctrl = np.array([16.4, 14.1, 13.4, 15.4, 14.0, 11.3])

        pmr4 = pm.permutation_test(test, ctrl, bs_stats.mean,
                                   bs_compare.difference, num_iterations=10000)

        self.assertAlmostEqual(
            pmr4,
            0.019,
            delta=0.005
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

    def test_randomized_permutation_ratio(self):
        test = np.array([126, 114, 132, 112, 94, 120])
        test_denom = np.ones(6)*10
        ctrl = np.array([164, 141, 134, 154, 140, 113])
        ctrl_denom = np.ones(6)*10

        pmr = pm.permutation_test(test, ctrl, bs_stats.mean,
                                   bs_compare.difference,
                                   test_denominator=test_denom,
                                   ctrl_denominator=ctrl_denom,
                                   num_iterations=10000)

        self.assertAlmostEqual(
            pmr,
            0.019,
            delta=0.002
        )

    def test_bootstrap_batch_size(self):
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

    def test_bootstrap_threads(self):
        mean = 100
        stdev = 10

        test = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl = np.random.normal(loc=mean, scale=stdev, size=5000)
        test = test * 1.1

        bsr = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                              bs_compare.percent_change)

        bsr_thread = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                                     bs_compare.percent_change,
                                     num_threads=10)
        self.assertAlmostEqual(
            bsr.value,
            bsr_thread.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_thread.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_thread.upper_bound,
            delta=.1
        )

        bsr = bs.bootstrap(test, bs_stats.mean)

        bsr_thread = bs.bootstrap(test, bs_stats.mean,
                                  num_threads=10)
        self.assertAlmostEqual(
            bsr.value,
            bsr_thread.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_thread.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_thread.upper_bound,
            delta=.1
        )

    def test_pivotal(self):
        mean = 100
        stdev = 10

        test = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl = np.random.normal(loc=mean, scale=stdev, size=5000)
        test = test * 1.1

        bsr = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                              bs_compare.percent_change)

        bsr_percent = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                                      bs_compare.percent_change,
                                      is_pivotal=False)
        self.assertAlmostEqual(
            bsr.value,
            bsr_percent.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_percent.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_percent.upper_bound,
            delta=.1
        )

        bsr = bs.bootstrap(test, bs_stats.mean)

        bsr_percent = bs.bootstrap(test, bs_stats.mean,
                                   num_threads=10)
        self.assertAlmostEqual(
            bsr.value,
            bsr_percent.value,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_percent.lower_bound,
            delta=.1
        )

        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_percent.upper_bound,
            delta=.1
        )

    def test_bootstrap_sparse(self):
        mean = 100
        stdev = 10

        samples = np.random.normal(loc=mean, scale=stdev, size=5000)
        samples_sp = sparse.csr_matrix(samples)

        bsr = bs.bootstrap(samples, bs_stats.mean)
        bsr_sp = bs.bootstrap(samples_sp, bs_stats.mean)

        self.assertAlmostEqual(bsr.value, 100, delta=2)
        self.assertAlmostEqual(bsr.upper_bound, 102, delta=2)
        self.assertAlmostEqual(bsr.lower_bound, 98, delta=2)

        self.assertAlmostEqual(bsr.value, bsr_sp.value, delta=.2)
        self.assertAlmostEqual(bsr.upper_bound, bsr_sp.upper_bound, delta=.2)
        self.assertAlmostEqual(bsr.lower_bound, bsr_sp.lower_bound, delta=.2)

        bsr = bs.bootstrap(samples, bs_stats.sum)
        bsr_sp = bs.bootstrap(samples, bs_stats.sum)

        self.assertAlmostEqual(bsr.value, bsr_sp.value, delta=bsr.value / 100.)
        self.assertAlmostEqual(bsr.upper_bound, bsr_sp.upper_bound,
                               delta=bsr.upper_bound / 100.)
        self.assertAlmostEqual(bsr.lower_bound, bsr_sp.lower_bound,
                               delta=bsr.lower_bound / 100.)

        bsr = bs.bootstrap(samples, bs_stats.std)
        bsr_sp = bs.bootstrap(samples, bs_stats.std)

        self.assertAlmostEqual(bsr.value, bsr_sp.value, delta=.2)
        self.assertAlmostEqual(bsr.upper_bound, bsr_sp.upper_bound, delta=.2)
        self.assertAlmostEqual(bsr.lower_bound, bsr_sp.lower_bound, delta=.2)

    def test_bootstrap_ab_sparse(self):
        mean = 100
        stdev = 10

        test = np.random.normal(loc=mean, scale=stdev, size=500)
        ctrl = np.random.normal(loc=mean, scale=stdev, size=5000)
        test = test * 1.1
        test_sp = sparse.csr_matrix(test)
        ctrl_sp = sparse.csr_matrix(ctrl)

        bsr = bs.bootstrap_ab(test, ctrl, bs_stats.mean,
                              bs_compare.percent_change)
        bsr_sp = bs.bootstrap_ab(test_sp, ctrl_sp, bs_stats.mean,
                                 bs_compare.percent_change)

        self.assertAlmostEqual(
            bsr.value,
            bsr_sp.value,
            delta=.1,
        )
        self.assertAlmostEqual(
            bsr.upper_bound,
            bsr_sp.upper_bound,
            delta=.1,
        )

        self.assertAlmostEqual(
            bsr.lower_bound,
            bsr_sp.lower_bound,
            delta=.1,
        )

    def test_t_dist(self):
        mean = 100
        stdev = 100

        sample_size = [250, 500, 1000, 2500, 3500, 5000, 8000, 10000]

        for i in sample_size:
            samples = np.random.normal(loc=mean, scale=stdev, size=i)
            bsr = bs.bootstrap(samples, stat_func=bs_stats.mean, alpha=0.05)

            mr = st.t.interval(1 - 0.05, len(samples) - 1, loc=np.mean(samples),
                               scale=st.sem(samples))

            self.assertAlmostEqual(
                bsr.lower_bound,
                mr[0],
                delta=mr[0] / 100.
            )

            self.assertAlmostEqual(
                bsr.upper_bound,
                mr[1],
                delta=mr[1] / 100.
            )
