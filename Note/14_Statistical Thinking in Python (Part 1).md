# Statistical Thinking in Python (Part 1)

## 1. Graphical exploratory data analysis
> * you should first explore your data by plotting them and computing simple summary statistics. 
> * This process, called exploratory data analysis, is a crucial first step in statistical analysis of data

### Introduction to exploratory data analysis
> #### Tukey's comments on EDA
> > * Exploratory data analysis is detective work.
> > * There is no excuse for failing to plot and look.
> > * The greatest value of a picture is that it forces us to notice what we never expected to see.
> > * It is important to understand what you can do before you learn how to measure how well you seem to have done it.

> #### Advantages of graphical EDA
> > * It often involves converting tabular data into graphical form.
> > * If done well, graphical representations can allow for more rapid interpretation of data.
> > * There is no excuse for neglecting to do graphical EDA.

### Plotting a histogram
> #### Plotting a histogram of iris data
> > * using matplotlib/seaborn's default settings. Recall that to specify the default seaborn style, you can use sns.set(), where sns is the alias that seaborn is imported as.
> > * Justin assigned his plotting statements (except for plt.show()) to the dummy variable _. This is to prevent unnecessary output from being displayed. It is not required for your solutions to these exercises, however it is good practice to use it

```python
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns


# Set default Seaborn style
sns.set()

# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)

# Show histogram
plt.show()
```

> #### Axis labels!
> > * Always label your axes

```python
# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')


# Show histogram
plt.show()
```
![217](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/217.png)

> #### Adjusting the number of bins in a histogram
> > * The histogram you just made had ten bins. This is the default of matplotlib.
> > * The "square root rule" is a commonly-used rule of thumb for choosing number of bins: choose the number of bins to be the square root of the number of samples

```python
# Import numpy
import numpy as np

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
plt.hist(versicolor_petal_length, bins=n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()
```
![218](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/218.png)

### Plotting all of your data: Bee swarm plots
> #### Bee swarm plot
> > * x-axis should contain each of the three species, and the y-axis the petal lengths. A data frame containing the data is in your namespace as df.

```python
# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x='species', y='petal length (cm)', data=df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length')

# Show the plot
plt.show()

```

![219](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/219.png)

> #### Interpreting a bee swarm plot
> > * I. virginica petals tend to be the longest, and I. setosa petals tend to be the shortest of the three species.

### Plotting all of your data: Empirical cumulative distribution functions
> #### Computing the ECDF
> > * write a function that takes as input a 1D array of data and then returns the x and y values of the ECDF.
> > * ECDFs are among the most important plots in statistical analysis.

```python
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n


    return x, y
```

> #### Plotting the ECDF

```python
y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle = 'none')

# Make the margins nice
plt.margins(0.02)

# Label the axes
_ = plt.xlabel('versicolor_petal_length')
_ = plt.ylabel('ECDF')


# Display the plot
plt.show()
``` 
![221](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/221.png)

![220](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/220.png)

> #### Comparison of ECDFs

```python
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Make nice margins
_ = plt.margins(0.02)

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
``` 
![222](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/222.png)

## 2. Quantitative exploratory data analysis
> * compute useful summary statistics, which serve to concisely describe salient features of a data set with a few numbers.

### Introduction to summary statistics: The sample mean and median
> #### Means and medians
> > * Mean: 
> > * Outliers: Data points whose value is far greater or less than most of the rest of the data
> > * Median: The middle value of a data set
> > * An outlier can significantly affect the value of the mean, but not the median.

> #### Computing means

```python
# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')
```

### Percentiles, outliers, and box plots
> #### Computing percentiles

```python
# Specify array of percentiles: percentiles
percentiles = [2.5, 25, 50, 75, 97.5]

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)
```
> #### Comparing percentiles to ECDF

```python
# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
plt.margins(0.02)
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot
_ = plt.show()
```
![223](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/223.png)

> #### Box-and-whisker plot
> > * Making a box plot for the petal lengths is unnecessary because the iris data set is not too large and the bee swarm plot works fine. 
> > * However, it is always good to get some practice

```python
# Create box plot with Seaborn's default settings
_ = sns.boxplot(x='species', y='petal length (cm)', data=df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')


# Show the plot
_ = plt.show()
```
![224](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/224.png)

### Variance and standard deviation
> #### Computing the variance
> > * Variance: The mean squared distance of the data from their mean, Informally, a measure of the spread of data

![225](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/225.png)

```python
# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results 
print(variance_explicit, variance_np) # 0.2164 0.2164
```
> #### The standard deviation and the variance
> > * by computing the standard deviation using np.std() and comparing it to what you get by computing the variance with np.var() and then computing the square root

```python
# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(np.sqrt(variance)) # 0.465188133985

# Print the standard deviation
print(np.std(versicolor_petal_length)) # 0.465188133985
```

### Covariance and Pearson correlation coefficient
> #### Scatter plots
> > * When you made bee swarm plots, box plots, and ECDF plots in previous exercises, you compared the petal lengths of different species of iris
> > * But what if you want to compare two properties of a single species? This is exactly what we will do in this exercise. We will make a scatter plot of the petal length and width measurements of Anderson's Iris versicolor flowers

```python
# Make a scatter plot
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')


# Set margins
_ = plt.margins(0.02)

# Label the axes
_ = plt.xlabel('versicolor_petal_length')
_ = plt.ylabel('versicolor_petal_width')

# Show the result
_ = plt.show()
```
![226](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/226.png)

> #### Variance and covariance by looking

![227](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/227.png)

![228](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/228.png)

![229](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/229.png)

> #### Computing the covariance

![230](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/230.png)

```python
# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)
```
![231](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/231.png)

> #### Computing the Pearson correlation coefficient
> > *  It is computed using the np.corrcoef() function. Like np.cov(), it takes two arrays as arguments and returns a 2D array. Entries [0,0] and [1,1] are necessarily equal to 1 (can you think about why?), and the value we are after is entry [0,1]

```python
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result
print(r) # 0.786668088523
```

## 3. Thinking probabilistically-- Discrete variables
> * Statistical inference rests upon probability
> * Because we can very rarely say anything meaningful with absolute certainty from data, we use probabilistic language to make quantitative statements about data.
> * It is an important first step in building the probabilistic language necessary to think statistically.

### Probabilistic logic and statistical inference
> #### What is the goal of statistical inference?
> > * To draw probabilistic conclusions about what we might expect if we collected the same data again.
> > * To draw actionable conclusions from data.
> > * To draw more general conclusions from relatively few data or observations.

> #### Why do we use the language of probability?
> > * Probability provides a measure of uncertainty.
> > * Data are almost never exactly the same when acquired again, and probability allows us to say how much we expect them to vary.

### Random number generators and hacker statistics
> #### Generating random numbers using the np.random module
> > * its simplest function, np.random.random() for a test spin. The function returns a random number between zero and one.


```python
# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()
```
![232](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/232.png)

> > * The histogram is almost exactly flat across the top, indicating that there is equal chance that a randomly-generated number is in any of the bins of the histogram.

> #### The np.random module and Bernoulli trials
> > * a Bernoulli trial as a flip of a possibly biased coin.
> > * each coin flip has a probability pp of landing heads (success) and probability 1−p1−p of landing tails (failure). 
> > * write a function to perform n Bernoulli trials, perform_bernoulli_trials(n, p), which returns the number of successes out of n Bernoulli trials, each of which has probability p of success.

```python
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success+=1

    return n_success

```

### Probability distributions and stories: The Binomial distribution
> #### Sampling out of the Binomial distribution
> > * The number r of successes in n Bernoulli trials with probability p of success, is Binomially distributed
> > * The number r of heads in 4 coin flips with probability 0.5 of heads, is Binomially distributed
> > * using np.random.binomial(). This is identical to the calculation you did in the last set of exercises using your custom-written perform_bernoulli_trials()

```python
# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(100, 0.05, size= 10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('the number of defaults out of 100 loans')
_ = plt.ylabel('CDF')

# Show the plot
_ = plt.show()
```
![233](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/233.png)

> #### Plotting the Binomial PMF
> > * Probability mass function
> > * The set of probabilities of discrete outcomes

![234](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/234.png)

![235](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/235.png)

### Poisson processes and the Poisson distribution
> #### Relationship between Binomial and Poisson distributions
> > * Poisson process : The timing of the next event is completely independent of when the previous event happened
> > * ex. Hit on a website during a given hour
> > * The number r of arrivals of a Poisson process in a given time interval with average rate of λ arrivals per interval is Poisson distributed.
> > * The number r of hits on a website in one hour with an average hit rate of 6 hits per hour is Poisson distributed.
> > * Poisson distribution = Limit of the Binomial distribution for low probability of success and large number of trials = That is, for rare events

```python
# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))
```
![236](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/236.png)

> > * The standard deviation of the Binomial distribution gets closer and closer to that of the Poisson distribution as the probability p gets lower and lower

## 4. Thinking probabilistically-- Continuous variables
> * In the last chapter, you learned about probability distributions of discrete variables. Now it is time to move on to continuous variables, such as those that can take on any fractional value

### Probability density functions
> #### Interpreting PDFs

![237](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/237.png)

> > * The probability is given by the area under the PDF, and there is more area to the left of 10 than to the right.

![238](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/238.png)

> #### Interpreting CDFs

![239](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/239.png)

### Introduction to the Normal distribution
> #### The Normal PDF
> > * Describes a continuous variable whose PDF has a single symmetric peak

![240](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/240.png)

![241](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/241.png)

```python
# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std3 = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)

# Make histograms
_ = plt.hist(samples_std1, normed=True, histtype='step', bins=100)
_ = plt.hist(samples_std3, normed=True, histtype='step', bins=100)
_ = plt.hist(samples_std10, normed=True, histtype='step', bins=100)


# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()
```
> > * how the different standard deviations result in PDFS of different widths. The peaks are all centered at the mean of 20.

![242](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/242.png)

> #### The Normal CDF

```python
# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)


# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make 2% margin
plt.margins(0.02)

# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()
```
> > * The CDFs all pass through the mean at the 50th percentile; the mean and median of a Normal distribution are equal. The width of the CDF varies with the standard deviation.

![243](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/243.png)

### The Normal distribution: Properties and warnings
> #### Are the Belmont Stakes results Normally distributed?
> > * Since 1926, the Belmont Stakes is a 1.5 mile-long race of 3-year old thoroughbred horses. 
> > * Secretariat ran the fastest Belmont Stakes in history in 1973. While that was the fastest year, 1970 was the slowest because of unusually wet and sloppy conditions. With these two outliers removed from the data set, 
> > * compute the mean and standard deviation of the Belmont winners' times.
> > * Sample out of a Normal distribution with this mean and standard deviation using the np.random.normal() function and plot a CDF. 
> > * Overlay the ECDF from the winning Belmont times. Are these close to Normally distributed?

```python
# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x, y = ecdf(belmont_no_outliers)
x_theor, y_theor = ecdf(samples)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()
```
> > * TThe theoretical CDF and the ECDF of the data suggest that the winning Belmont times are, indeed, Normally distributed

![244](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/244.png)

> #### What are the chances of a horse matching or beating Secretariat's record?
> > * Assume that the Belmont winners' times are Normally distributed (with the 1970 and 1973 years removed), 
> > * what is the probability that the winner of a given Belmont Stakes will run it as fast or faster than Secretariat?
> > * We had to take a million samples because the probability of a fast time is very low and we had to be sure to sample enough. We get that there is only a 0.06% chance of a horse running the Belmont as fast as Secretariat.

```python
# Compute mean and standard deviation: mu, sigma
# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)  

# Probability of besting Secretariat: 0.000635
```


### The Exponential distribution
> * The waiting time between arrivals of a Poisson process is Exponentially distributed

![245](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/245.png)

> #### If you have a story, you can simulate it!
> > * How long must we wait to see both a no-hitter and a batter hit the cycle? 
> > * The idea is that we have to wait some time for the no-hitter, and then after the no-hitter, we have to wait for hitting the cycle.
> > * Stated another way, what is the total waiting time for the arrival of two different Poisson processes? 
> > * The total waiting time is the time waited for the no-hitter, plus the time waited for the hitting the cycle.
> > * will write a function to sample out of the distribution described by this story.

```python
def successive_poisson(tau1, tau2, size=1):
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)

    return t1 + t2
```

> #### Distribution of no-hitters and cycles
> > * The mean waiting time for a no-hitter is 764 games, and the mean waiting time for hitting the cycle is 715 games.

```python
# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
_ = plt.hist(waiting_times, bins=100, normed=True, histtype='step')


# Label axes
_ = plt.ylabel('PDF')
_ = plt.xlabel('time (days)')


# Show the plot
plt.show()
```

![246](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/246.png)