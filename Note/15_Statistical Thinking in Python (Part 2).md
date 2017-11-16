# Statistical Thinking in Python (Part 2)

## 1. Parameter estimation by optimization
> * A probability distribution that describes your data has parameters. 
> * So, a major goal of statistical inference is to estimate the values of these parameters, which allows us to concisely and unambiguously describe our data and draw conclusions from it.

### Optimal parameters
> #### How often do we get no-hitters?
> > * The number of games played between each no-hitter in the modern era (1901-2015) of Major League Baseball is stored in the array nohitter_times
> > * If you assume that no-hitters are described as a Poisson process, then the time between no-hitters is Exponentially distributed.
> > * the Exponential distribution has a single parameter, which we will call \tau
> > * The value of the parameter ττ that makes the exponential distribution best match the data is the mean interval time (where time is in units of number of games) between no-hitters.

```python
# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()
```

![247](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/247.png)

> #### Do the data follow our story?
> > * Create an ECDF of the real data. 
> > * Overlay the theoretical CDF with the ECDF from the data. This helps you to verify that the Exponential distribution describes the observed data.

```python
# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()
```

![248](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/248.png)

> #### How is this parameter optimal?
> > * Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter

```python
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, 10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, 10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()
```

![249](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/249.png)

### Linear regression by least squares
> #### EDA of literacy/fertility data
> > * You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8.
> > * It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.

```python
# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility)) # 0.804132402682
```

![250](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/250.png)

> #### Linear regression
> > * f=ai+b, where a is the slope and b is the intercept.
> > * We can find the best fit line using np.polyfit().

```python
# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, 1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()
```

![251](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/251.png)

> #### How is it optimal?
> > * It is optimizing the sum of the squares of the residuals, also known as RSS (for residual sum of squares)
> > * np.sum((y_data - a * x_data - b)**2)
> > * Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals, is the same value you got when performing the regression.

```python
# Specify slopes to consider: a_vals
a_vals = np.linspace(0, 0.1, 200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()
```

![252](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/252.png)

### The importance of EDA: Anscombe's quartet
> * Do graphical EDA first
> * EDA provides a good starting point for planning the rest of your analysis.

![253](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/253.png)

> #### Linear regression on appropriate Anscombe data
> > * The Anscombe data are stored in the arrays x and y.

```python
# Perform linear regression: a, b
a, b = np.polyfit(x, y, 1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(x_theor, y_theor, marker='.', linestyle='none')

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()
```

![254](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/254.png)

> #### Linear regression on all Anscombe data
> > * anscombe_x = [x1, x2, x3, x4] and anscombe_y = [y1, y2, y3, y4], where, for example, x2 and y2 are the xx and yy values for the second Anscombe data set
> > * They all have the same slope and intercept

```python
for x, y in zip(anscombe_x, anscombe_y):
    # Compute the slope and intercept: a, b
    a, b = np.polyfit(x, y, 1)

    # Print the result
    print('slope:', a, 'intercept:', b)
```

![255](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/255.png)

## 2. Bootstrap confidence intervals
> * can we use only the data we actually have to get close to the same result as an infinitude of experiments?
> * The answer is yes! The technique to do it is aptly called bootstrapping.

### Generating bootstrap replicates
> #### Getting the terminology down
> > * Bootstrapping: The use of resampled data to perform statistical inference
> > * Bootstrap sample: A resampled array of the data
> > * Bootstrap replicate: A statistic computed from a resampled array

> #### Bootstrapping by hand
> > * imagine you have a data set that has only three points, [-1, 0, 1]. 
> > * How many unique bootstrap samples can be drawn (e.g., [-1, 0, 1] and [1, 0, -1] are unique), and what is the maximum mean you can get from a bootstrap sample?
> > * There are 27 unique samples, and the maximum mean is 1.

> #### Visualizing bootstrap samples
> > * generate bootstrap samples from the set of annual rainfall data measured at the Sheffield Weather Station in the UK from 1883 to 2015. The data are stored in the NumPy array rainfall

```python
for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()
```

![256](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/256.png)

### Bootstrap confidence intervals
> * If we repeated measurements over and over again, p% of the observed values would lie within the p% confidence interval.

> #### Generating many bootstrap replicates
> > * draw_bs_reps(data, func, size=1), which generates many bootstrap replicates from the data set. 
> > * This function will come in handy for you again and again as you compute confidence intervals and later when you do hypothesis tests.

```python
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
```

> #### Bootstrap replicates of the mean and the SEM
> > * The standard deviation of this distribution, called the standard error of the mean, or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set, sem = np.std(data) / np.sqrt(len(data))
> > * the SEM we got from the known expression and the bootstrap replicates is the same and the distribution of the bootstrap replicates of the mean is Normal.

```python
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()
```
![258](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/258.png)

![257](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/257.png)

> #### Confidence intervals of rainfall data
> > * Using your bootstrap replicates you just generated to compute the 95% confidence interval. 
> > * That is, give the 2.5th and 97.5th percentile of your bootstrap replicates stored as bs_replicates.

![259](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/259.png)

> #### Bootstrap replicates of other statistics

```python
# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, 10000)

# Put the variance in units of square centimeters
bs_replicates = bs_replicates / 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()
```
> #### Confidence interval on the rate of no-hitters
> > *  Generate 10,000 bootstrap replicates of the optimal parameter ττ. 
> > * Plot a histogram of your replicates and report a 95% confidence interval.
> > * This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games

```python
# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times, np.mean, 10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates, [2.5, 97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')
```
![260](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/260.png)

![261](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/261.png)

### Pairs bootstrap
> #### A function to do pairs bootstrap
> > * pairs bootstrap involves resampling pairs of data. 
> > * Each collection of pairs fit with a line, in this case using np.polyfit()

```python
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps
```

> #### Pairs bootstrap of literacy/fertility data

```python
# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, 1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5])) # [ 0.04378061  0.0551616 ]

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()
```
![262](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/262.png)

> #### Plotting bootstrap regressions
> > * A nice way to visualize the variability we might expect in a linear regression is to plot the line you would get from each bootstrap replicate of the slope and intercep
> > * You now have some serious chops for parameter estimation. Let's move on to hypothesis testing!

```python
# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()
```
![263](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/263.png)


## 3. Introduction to hypothesis testing
> * You now know how to define and estimate parameters given a model. 
> * But the question remains: how reasonable is it to observe your data if a model is true? 
> * This question is addressed by hypothesis tests. 

### Formulating and simulating a hypothesis
> * Hypothesis testing: Assessment of how reasonable the observed data are assuming a hypothesis is true
> * Null hypothesis: Another name for the hypothesis you are testing
> * Permutation: Random reordering of entries in an array

> #### Generating a permutation sample
> > * permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions. 
> > * This is often a hypothesis you want to test, write a function to generate a permutation sample from two data sets.

```python
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
```

> #### Visualizing permutation sampling
> > * Notice that the permutation samples ECDFs overlap and give a purple haze. 
> > * None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not commensurate with the data. 
> > * July and November rainfall are not identically distributed

```python
for i in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_july, rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_july)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()
```
![264](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/264.png)


### Test statistics and p-values
> * Test statistic: A single number that can be computed from observed data and from data you simulate under the null hypothesis
> * p-value: The probability of obtaining a value of your test statistic that is at least as extreme as what was observed, under the assumption the null hypothesis is true
> * NOT the probability that the null hypothesis is true
> * Statistical significance: Determined by the smallness of a p-value
> * Null hypothesis significance testing (NHST): Another name for what we are doing in this chapter
> * statistical significance ≠ practical significance

> #### Test statistics
> > * be pertinent to the question you are seeking to answer in your hypothesis test.

> #### What is a p-value?
> > * the probability of observing a test statistic equally or more extreme than the one you observed, assuming the hypothesis you are testing is true.

> #### Generating permutation replicates
> > * a permutation replicate is a single value of a statistic computed from a permutation sample
> > * As the draw_bs_reps() function you wrote in chapter 2 is useful for you to generate bootstrap replicates, it is useful to have a similar function, draw_perm_reps(), to generate permutation replicates. 

```python
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates
```

> #### Look before you leap: EDA before hypothesis testing
> > * Frog A, the adult, has three or four very hard strikes
> > * Frog B, the juvenile, has a couple weak ones. 
> > * However, it is possible that with only 20 samples it might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.

```python
# Make bee swarm plot
_ = sns.swarmplot(data=df, x='ID', y='impact_force')

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()

```
![265](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/265.png)

> #### Permutation test on frog data
> > * compute the probability of getting at least a 0.29 N(Newtons) difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. 
> > * use a permutation test with a test statistic of the difference of means to test this hypothesis.
> > * the data has been stored in the arrays force_a and force_b.

```python
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p) # 0.0063

```
> > * The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the experiment if frogs were exactly the same. 
> > * A p-value below 0.01 is typically said to be "statistically significant,", 
> > * but: warning! warning! warning! You have computed a p-value; it is a number. I encourage you not to distill it to a yes-or-no phrase. 
> > * p = 0.006 and p = 0.000000006 are both said to be "statistically significant," but they are definitely not the same!

### Bootstrap hypothesis tests
> #### Pipeline for hypothesis testing
> > * Clearly state the null hypothesis
> > * Define your test statistic
> > * Generate many sets of simulated data assuming the null hypothesis is true
> > * Compute the test statistic for each simulated data set
> > * The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data

> #### A one-sample bootstrap hypothesis test
> > * Unfortunately, you do not have Frog C's impact forces available, but you know they have a mean of 0.55 N.
> > * Translate the impact forces of Frog B such that its mean is 0.55 N.

```python
# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p) # 0.0046

```
> > * The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.

> #### A bootstrap test for identical distributions
> > * we looked at a one-sample test, but we can do two sample tests.
> > We concatenate the arrays, generate a bootstrap sample from it, and take the first n1 entries of the bootstrap sample as belonging to the first data set and the last n2 as belonging to the second.
> > * The two arrays are available to you as force_a and force_b

```python
# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Concatenate forces: forces_concat
forces_concat = np.concatenate((force_a, force_b))

# Initialize bootstrap replicates: bs_replicates
bs_replicates = np.empty(10000)

for i in range(10000):
    # Generate bootstrap sample d 
    bs_sample = np.random.choice(forces_concat, size=len(forces_concat))
    
    # Compute replicate
    bs_replicates[i] = diff_of_means(bs_sample[:len(force_a)],
                                     bs_sample[len(force_a):])

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / len(bs_replicates)
print('p-value =', p) # 0.0055

```
> > * You may remember that we got p = 0.0063 from the permutation test, and here we got p = 0.0055. These are very close, and indeed the tests are testing the same thing. 
> > * the permutation test exactly simulates the null hypothesis that the data come from the same distribution
> > * the bootstrap test approximately simulates it. 
> > * As we will see, though, the bootstrap hypothesis test, while approximate, is more versatile.

> #### A two-sample bootstrap hypothesis test for difference of means.
> > * Testing the hypothesis that two samples have the same distribution may be done with a bootstrap test, but a permutation test is preferred because it is more accurate (exact, in fact)
> > * To do the two-sample bootstrap test, we shift both arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. 
> > * The objects forces_concat and empirical_diff_means are already in your namespace.

```python
# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force 

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / len(bs_replicates)
print('p-value =', p) # 0.0043

```
> > * the more forgiving hypothesis, only that the means are equal as opposed to having identical distributions, gives a higher p-value. Again, 
> > * it is important to carefully think about what question you want to ask. Are you only interested in the mean impact force, or the distribution of impact forces?

## 4. Hypothesis test examples
> * hypothesis testing can be a bit tricky. 
> * You need to define the null hypothesis, figure out how to simulate it, 
> * and define clearly what it means to be "more extreme" in order to compute the p-value. 

### A/B testing
> * A/B test: Used by organizations to see if a strategy change gives a be er result
> * Null hypothesis of an A/B test: The test statistic is impervious to the change

### Test of correlation
> #### Simulating a null hypothesis concerning correlation
> > * The observed correlation between female illiteracy and fertility in the data set of 162 countries may just be by chance; 
> > * the fertility of a given country may actually be totally independent of its illiteracy.
> > * To do the test, you need to simulate the data assuming the null hypothesis is true. which is the best way to to do it?
> > * Do a permutation test: Permute the illiteracy values but leave the fertility values fixed to generate a new set of (illiteracy, fertility) data.

> #### Hypothesis test on Pearson correlation
> > * permute the illiteracy values but leave the fertility values fixed.
> > * This simulates the hypothesis that they are totally independent of each other. 
> > * For each permutation, compute the Pearson correlation coefficient and assess how many of your permutation replicates have a Pearson correlation coefficient greater than the observed one.

```python
# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
print('p-val =', p) # 0.0

```

> > * got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one

## 5. Putting it all together: a case study
> * Our data source: Peter and Rosemary Grant40 Years of Evolution: Darwin's Finches on Daphne Major IslandPrinceton University Press, 2014.
> * you will spend this chapter with their data, and witness first hand, through data, evolution in action. It's an exhilarating way to end the course!

![266](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/266.png)

### Finch beaks and the need for statistics
> #### EDA of beak depths of Darwin's finches
> > * study how the beak depth (the distance, top to bottom, of a closed beak) of the finch species Geospiza scandens has changed over time. 

```python
# Create bee swarm plot
_ = sns.swarmplot(data=df, x='year', y='beak_depth')

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

```
![267](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/267.png)

> > * It is kind of hard to see if there is a clear difference between the 1975 and 2012 data set. 
> > * Eyeballing it, it appears as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.

> #### ECDFs of beak depths
> > * For your convenience, the beak depths for the respective years has been stored in the NumPy arrays bd_1975 and bd_2012

```python
# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()
```

![268](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/268.png)

> > * The differences are much clearer in the ECDF. The mean is larger in the 2012 data, and the variance does appear larger as well.

> #### Parameter estimates of beak depths
> > * Estimate the difference of the mean beak depth of the G. scandens samples from 1975 and 2012 and report a 95% confidence interval

```python
# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

#   difference of means = 0.226220472441 mm
#   95% confidence interval = [ 0.05633521  0.39190544] mm
```
> #### Hypothesis test: Are beaks deeper in 2012?
> > * Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of G. scandens on Daphne Major have gotten deeper. 
> > * But is it possible that this effect is just due to random chance?
> > *  In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?
> > * The hypothesis we are testing is not that the beak depths come from the same distribution.
> > * The hypothesis is that the means are equal. 
> > * To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.

```python
# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, 10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p) # p = 0.0034
```
> > * get a p-value of 0.0034, which suggests that there is a statistically significant difference.
> > * In the previous exercise, you got a difference of 0.2 mm between the means.
> > * You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards.

### Variation of beak shapes
> #### EDA of beak length and depth
> > * The beak length data are stored as bl_1975 and bl_2012, again with units of millimeters (mm). You still have the beak depth data stored in bd_1975 and bd_2012

```python
# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()
```

![269](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/269.png)

> > * we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. 
> > * If anything, they got a bit shorter, since the red dots are to the left of the blue dots. 
> > * So, it does not look like the beaks kept the same shape; they became shorter and deeper.

> #### Linear regressions
> > * Perform a linear regression for both the 1975 and 2012 data.
> > * Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.

```python
# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = \
        draw_bs_pairs_linreg(bl_1975, bd_1975, 1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = \
        draw_bs_pairs_linreg(bl_2012, bd_2012, 1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])

intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)
```

![270](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/270.png)

> > * It looks like they have the same slope, but different intercepts.

> #### Displaying the linear regression results
> > * take the first 100 bootstrap samples (stored in bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, and bs_intercept_reps_2012) 

```python
# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()
```

![271](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/271.png)

> #### Beak length to depth ratio
> > * The slope was the same in 1975 and 2012, suggesting that for every millimeter gained in beak length, the birds gained about half a millimeter in depth in both years. 
> > * However, if we are interested in the shape of the beak, we want to compare the ratio of beak length to beak depth. 
> > * the data are stored in bd_1975, bd_2012, bl_1975, and bl_2012

```python
# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, 10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)
```

![272](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/272.png)

> #### How different is the ratio?
> > * showed that the mean beak length to depth ratio was 1.58 in 1975 and 1.47 in 2012. 
> > * The low end of the 1975 99% confidence interval was 1.56 mm and the high end of the 99% confidence interval in 2012 was 1.49 mm.
> > * In addition to these results, what would you say about the ratio of beak length to depth?
> > * The mean beak length-to-depth ratio decreased by about 0.1, or 7%, from 1975 to 2012. The 99% confidence intervals are not even close to overlapping, so this is a real change. The beak shape changed.

### Calculation of heritability
> * Heredity: The tendency for parental traits to be inherited by offspring

> #### EDA of heritability
> > * The array bd_parent_scandens contains the average beak depth (in mm) of two parents of the species G. scandens. 
> > * The array bd_offspring_scandens contains the average beak depth of the offspring of the respective parents. 
> > * The arrays bd_parent_fortis and bd_offspring_fortis contain the same information about measurements from G. fortis birds

```python
# Make scatter plots
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)

# Set margins
plt.margins(0.02)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()
```

![273](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/273.png)

> > * It appears as though there is a stronger correlation in G. fortis than in G. scandens. This suggests that beak depth is more strongly inherited in G. fortis.

> #### Correlation of offspring and parental data
> > * In an effort to quantify the correlation between offspring and parent beak depths, 
> > * we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring.
> > * You have already written a function to do pairs bootstrap to get estimates for parameters derived from linear regression. 
> > * Your task in this exercise is to modify that function to make a new function with call signature draw_bs_pairs(x, y, func, size=1) that performs pairs bootstrap and computes a single statistic on the pairs samples defined by func(bs_x, bs_y). In the next exercise, you will use pearson_r for func.

```python
def draw_bs_pairs(x, y,func ,size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = func(bs_x, bs_y)

    return bs_slope_reps, bs_intercept_reps
```

> #### Pearson correlation of offspring and parental data
> > *  the data are stored in bd_parent_scandens, bd_offspring_scandens, bd_parent_fortis, and bd_offspring_fortis.

```python
# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, 1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, 1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)
```
![274](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/274.png)

> > * It is clear from the confidence intervals that beak depth of the offspring of G. fortis parents is more strongly correlated with their offspring than their G. scandens counterparts.