# bootstrapped - confidence intervals made easy

**bootstrapped** is a Python library that allows you to build confidence intervals from data. This is useful in a variety of contexts - including during ad-hoc a/b test analysis.

## Motivating Example - A/B Test
Imagine we own a website and think changing the color of a 'subscribe' button will improve signups. One method to measure the improvement is to conduct an A/B test where we show 50% of people the old version and 50% of the people the new version. We can use the bootstrap to understand how much the button color improves responses and give us the error bars associated with the test - this will give us lower and upper bounds on how good we should expect the change to be!

## The Gist - Mean of a Sample
Given a sample of data - we can generate a bunch of new samples by 're-sampling' from what we have gathered. We calculate the mean for each generated sample. We can use the means from the generated samples to understand the variation in the larger population and can construct error bars for the true mean.

## bootstrapped - Benefits
 * Efficient computation of confidence intervals
 * Functions to handle single populations and a/b tests
 * Functions to understand [statistical power](https://en.wikipedia.org/wiki/Statistical_power) 
 * Multithreaded support to speed-up bootstrap computations

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

print(bs.bootstrap(samples, stat_func=bs_stats.mean))
>> 100.08  (99.46, 100.69)

print(bs.bootstrap(samples, stat_func=bs_stats.std))
>> 9.49  (9.92, 10.36)
```
#### Extended Examples
* [Bootstrap Intro](https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_intro.ipynb)
* [Bootstrap A/B Testing](https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_ab_testing.ipynb)
* More notebooks can be found in the [examples/](https://github.com/facebookincubator/bootstrapped/tree/master/examples) directory 

## Requirements
**bootstrapped** requires numpy. The power analysis functions require matplotlib and pandas. 

## Installation
```bash
# clone bootstrapped 
cd bootstrapped 
pip install -r requirements.txt 
python setup.py install
```

## How bootstrapped works
**bootstrapped** provides pivotal (aka empirical) based confidence intervals based on bootstrap re-sampling with replacement. The percentile method is also available.

For more information please see:

1. [Bootstrap confidence intervals](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf) (good intro)
2. [An introduction to Bootstrap Methods](http://www.stat-athens.aueb.gr/~karlis/lefkada/boot.pdf)
3. [The Bootstrap, Advanced Data Analysis](http://www.stat.cmu.edu/~cshalizi/402/lectures/08-bootstrap/lecture-08.pdf)
4. [When the bootstrap dosen't work](http://notstatschat.tumblr.com/post/156650638586/when-the-bootstrap-doesnt-work)
5. (book) [An Introduction to the Bootstrap](https://www.amazon.com/Introduction-Bootstrap-Monographs-Statistics-Probability/dp/0412042312/)
6. (book) [Bootstrap Methods and their Application](https://www.amazon.com/Bootstrap-Application-Statistical-Probabilistic-Mathematics-ebook/dp/B00D2WQ02U/)

See the CONTRIBUTING file for how to help out.

#### Contributors
Spencer Beecher, Don van der Drift, David Martin, Lindsay Vass, Sergey Goder, Benedict Lim, and Matt Langner.

Special thanks to Eytan Bakshy.

## License
**bootstrapped** is BSD-licensed. We also provide an additional patent grant.
