# bootstrapped - confidence intervals made easy

**bootstrapped** is a python library that allows you to build confidence intervals from data. This is useful in a variety of contexts - including during ad-hoc a/b test analysis.

## bootstrapped - Benefits
 * Efficient computation of percentile based confidence intervals
 * Functions to handle single populations and a/b test scenarios
 * Functions to understand [statistical power](https://en.wikipedia.org/wiki/Statistical_power) 

## Example Usage
```python
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

mean = 100
stdev = 10

population = np.random.normal(loc=mean, scale=stdev, size=50000)

# take 1k 'samples' from the larger population
samples = population[:1000]

print bs.bootstrap(samples, stat_func=bs_stats.mean)
>> BootstrapResults(lower_bound=99.46, value=100.08, upper_bound=100.69)

print bs.bootstrap(samples, stat_func=bs_stats.std)
>> BootstrapResults(lower_bound=9.49, value=9.92, upper_bound=10.36)
```
#### Extended Examples
* [Bootstrap Intro](https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_intro.ipynb)
* [Bootstrap A/B Testing](https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_ab_testing.ipynb)
* More notebooks can be found in the [examples/](https://github.com/facebookincubator/bootstrapped/tree/master/examples) diectory

## Requirements
**bootstrapped** requires numpy and pandas. The power analysis plotting function requires matplotlib. statsmodels is used in some of the examples.

## Installation
```bash
# clone bootstrapped 
cd bootstrapped 
pip install -r requirements.txt 
python setup.py install
```

## How bootstrapped works
tldr - Percentile based confidence intervals based on bootstrap re-sampling with replacement.

Bootstrapped generates confidence intervals given input data by:
* Generating a large number of samples from the input (re-sampling)
* For each re-sample, calculate the mean (or whatever statistic you care about)
* Of these results, calculate the 2.5th and 97.5 percentiles (default range)
* Use this as the 95% confidence interval

For more information please see:

1. [Bootstrap confidence intervals](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf) (good intro)
2. [An introduction to Bootstrap Methods](http://www.stat-athens.aueb.gr/~karlis/lefkada/boot.pdf)
3. [When the bootstrap dosen't work](http://notstatschat.tumblr.com/post/156650638586/when-the-bootstrap-doesnt-work)
4. (book) [An Introduction to the Bootstrap](https://www.amazon.com/Introduction-Bootstrap-Monographs-Statistics-Probability/dp/0412042312/)
5. (book) [Bootstrap Methods and their Application](https://www.amazon.com/Bootstrap-Application-Statistical-Probabilistic-Mathematics-ebook/dp/B00D2WQ02U/)

See the CONTRIBUTING file for how to help out.

#### Contributors
Spencer Beecher, Don van der Drift, David Martin, Lindsay Vass, Sergey Goder, Benedict Lim, and Matt Langner.

## License
**bootstrapped** is BSD-licensed. We also provide an additional patent grant.
