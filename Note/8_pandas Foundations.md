# pandas Foundations

## 1. Data ingestion & inspection
### Review of pandas DataFrames
> #### Inspecting your data
> use the DataFrame methods .head() and .tail() to view the first few and last few rows of a DataFrame
> 
> #### DataFrame data types
> use df.info() to determine information about the total count of non-null entries and infer the total count of 'null' entries, which likely indicates missing data
> 
> #### NumPy and pandas working together
> use the DataFrame attribute .values to represent a DataFrame df as a NumPy array.

```python
# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
print(type(np_vals), type(np_vals_log10))
print(type(df), type(df_log10))
```

### Building DataFrames from scratch
> #### Zip lists to build a DataFrame
> use the list of tuples to construct a dictionary, and then use that dictionary to construct a DataFrame. In doing so, you'll make use of the list(), zip(), dict() and pd.DataFrame()

```python
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
```

> #### Labeling your data
> use the DataFrame attribute df.columns to view and assign new string labels to columns in a pandas DataFrame.

```python
# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
```

> #### Building DataFrames with broadcasting
> can implicitly use 'broadcasting', a feature of NumPy(讓同一行都是同個值)

```python
# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)
```

### Importing & exporting data
> #### Reading a flat file
> using the pandas function read_csv()
> > names keyword input parameter, set equal to a list of new column labels. 
> > 
> > set header=0 to rename the column labels.

```python
# Read in the file: df1
df1 = pd.read_csv('world_population.csv')

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('world_population.csv', header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)
```

> #### Delimiters, headers, and extensions
> Not all data files are clean and tidy. Pandas provides methods for reading those not-so-perfect data files that you encounter far too often
> 
> this file has three aspects that may cause trouble for lesser tools:
> > * multiple header lines, 
> > * comment records (rows) interleaved throughout the data rows, 
> > * tab delimiters instead of commas.
>  
> Using the keyword arguments delimiter=' ', header=3 and comment='#'

```python
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)
```

### Plotting with pandas
> #### Plotting series using pandas
> .plot() method makes calls to matplotlib to construct the plots

```python
# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()
```
> #### Plotting DataFrames
> use a list of column names passed into the DataFrame df[column_list] to limit plotting to just one column

```pyhton
# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()
```


## 2. Exploratory data analysis
### Visual exploratory data analysis
> #### pandas line plots

```python
# Create a list of y-axis column names: y_columns
y_columns = ['AAPL','IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()
```

> #### pandas scatter plots
> > * using the kind='scatter' keyword argument.
> > * Scatter plots require that the x and y columns be chosen by specifying the x and y parameters inside .plot().
> > * Scatter plots also take an s keyword argument to provide the radius of each circle to plot in pixels.

```python
# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()
```

> #### pandas box plots
> plot multiple columns of data in a single figure, making plots that share the same x and y axes, there are cases where two columns cannot be plotted together because their units do not match. 
> 
> The .plot() method can generate subplots for each column being plotted. Here, each plot will be scaled independently.

```python
# Make a list of the column names to be plotted: cols
cols = ['weight','mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display the plot
plt.show()
```

> #### pandas hist, pdf and cdf
> .hist() method to not only generate histograms, but also plots of probability density functions (PDFs) and cumulative density functions (CDFs).
> > * when plotting the PDF, you need to specify normed=True in your call to .hist()
> > * when plotting the CDF, you need to specify cumulative=True in addition to normed=True.

```python
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
plt.show()
```

### Statistical exploratory data analysis
> #### Bachelor's degrees awarded to women
>  use the .mean() method with the keyword argument axis='columns'
> 
>  The DataFrame has been pre-loaded for you as df with the index set to 'Year'

```python
# Print the minimum value of the Engineering column
print(df['Engineering'].min())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()
```
> #### Median vs mean
> investigate the mean, median, and max fare prices paid by passengers on the Titanic and generate a box plot of the fare prices. 

```python
# Print summary statistics of the fare column with .describe()
print(df['fare'].describe())

# Generate a box plot of the fare column
df['fare'].plot(kind='box')

# Show the plot
plt.show()
```

> #### Quantiles
>  * determine the number of countries reported in 2015. There are a total of 206 unique countries in the entire dataset. 
> * Then, you will compute the 5th and 95th percentiles of life expectancy over the entire dataset. 
> * Finally, you will make a box plot of life expectancy every 50 years from 1800 to 2000. 
> * Notice the large change in the distributions over this period.

```python
# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()
```

> #### Standard deviation of temperature
> use the mean and standard deviation to explore differences in temperature distributions in Pittsburgh in 2013

```python
# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())
```

### Separating populations with Boolean indexing
> #### Separate and summarize
>  * compute the means and standard deviations of all columns in the full automobile dataset. 
>  * Next, you'll compute the same quantities for just the US population 
>  * subtract the global values from the US values

```python
# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df.loc[df['origin'] == 'US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)
```

> #### Separate and plot
> use Boolean filtering and generate box plots of the fare prices for each of the three passenger classes

```python
# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()

```

## 3. Time series in pandas
### Indexing pandas time series
> #### Reading and slicing times
>  we have read in the same data file using three different approaches:
>> * df1 = pd.read_csv(filename)
>> * df2 = pd.read_csv(filename, parse_dates=['Date'])
>> * df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)
>
> #### Creating and using a DatetimeIndex
>  if passed the list of strings ['2015-01-01 091234','2015-01-01 091234'] and a format specification variable, such as format='%Y-%m-%d %H%M%S, pandas will parse the string into the proper datetime elements and build the datetime objects.

```python
# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list,format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)
```
> #### Partial string indexing and slicing

```python
# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15' : '2010-12-31']
```
> #### Reindexing the Index
> * Reindexing is useful in preparation for adding or otherwise combining two time series data sets. 
> * To reindex the data, we provide a new index and ask pandas to try and match the old data to the new index. 
> * If data is unavailable for one of the new index dates or times, you must tell pandas how to fill it in. 
> * Otherwise, pandas will fill with NaN by default.

```python
# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index,method='ffill')

# Combine ts1 + ts2: sum12
sum12 = ts1 + ts2

# Combine ts1 + ts3: sum13
sum13 = ts1 + ts3

# Combine ts1 + ts4: sum14
sum14 = ts1 + ts4
```

### Resampling pandas time series
> #### Resampling and frequency
> * When downsampling or upsampling, the syntax is similar, but the methods called are different. Both use the concept of 'method chaining' - df.method1().method2().method3()
> * One approach, for instance, could be to take the mean, as in df.resample('D').mean().

```python
# Downsample to 6 hour data and aggregate by mean: df1
df1 = df['Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df['Temperature'].resample('D').count()
```

> #### Separating and resampling
>  you can resample in different ways on different subsets of your data

```python
# Extract temperature data for August: august
august = df.loc['August 2010', 'Temperature']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df.loc['February 2010', 'Temperature']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()
```

> #### Rolling mean and frequency
> *  Rolling means (or moving averages) are generally used to smooth out short-term fluctuations in time series data and highlight long-term trends.
> * use the .rolling() method, you must always use method chaining, first calling .rolling() and then chaining an aggregation method after it. For example, with a Series hourly_data, hourly_data.rolling(window=24).mean()

```python
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()
```

### Manipulating pandas time series
> #### Method chaining and filtering

```python
# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

```

> #### Missing values and interpolation
> The problem is that one of the data sets is missing data at some of the times. 
> > * The pre-loaded data ts1 has value for all times, 
> > * the data set ts2 does not: it is missing data for the weekends.

```python
# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1- ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())
```
> #### Time zones and conversion

```python
# Buid a Boolean mask to filter out all the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')
```

### Visualizing pandas time series

> #### Plotting time series, datetime indexing
> Pandas handles datetimes not only in your data, but also in your plotting.

```python
# Plot the raw data before setting the datetime index
df.plot()
plt.show()

# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)

# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()
```

> #### Plotting date ranges, partial indexing
> one of the most convenient is partial string indexing and slicing

```python
# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()
```


## 4. Case Study - Sunlight in Austin
### Reading and cleaning the data
> #### Reading in a data file

```python
# Import pandas
import pandas as pd

# Read in the data file: df
df = pd.read_csv('data.csv')

# Print the output of df.head()
print(df.head())

# Read in the data file with header=None: df_headers
df_headers = pd.read_csv('data.csv', header=None)

# Print the output of df_headers.head()
print(df_headers.head())

```
> #### Re-assigning column names

```python
# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(',')

# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list

# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop, axis='columns')

# Print the output of df_dropped.head()
print(df_dropped.head())
```
> #### Cleaning and tidying datetime data
> to clean up the date and Time columns and combine them into a datetime collection to be used as the Index

```python
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date']+df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())
```
> #### Cleaning the numeric columns
> The numeric columns contain missing values labeled as 'M'. In this exercise, your job is to transform these columns such that they contain only numeric values and interpret missing data as NaN

```python
# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')
```


### Statistical exploratory data analysis
> #### Signal min, max, median

```python
# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())
```

> #### Signal variance
> > * Your job is to first resample df_clean and df_climate by day and aggregate the mean temperatures. 
> > * You will then extract the temperature related columns from each - 'dry_bulb_faren' in df_clean, and 'Temperature' in df_climate - as NumPy arrays and compute the difference.

```python
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())
```

> #### Sunny or cloudy
> > * use Boolean selection to filter out sunny and overcast days
> > * then compute the difference of the mean daily maximum temperatures between each type of day.

```python
# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean['sky_condition']=='CLR']

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())

```

### Visual exploratory data analysis
> #### Weekly average temperature and visibility
> > * Is there a correlation between temperature and visibility?
> > * compute the Pearson correlation coefficient using .corr(). The Pearson correlation coefficient, known also as Pearson's r, ranges from -1 (indicating total negative linear correlation) to 1 (indicating total positive linear correlation). A value close to 1 here would indicate that there is a strong correlation between temperature and visibility.

```python
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility', 'dry_bulb_faren']].resample('W').mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()
```

> #### Daily hours of clear sky
> > use a box plot to visualize the fraction of days that are sunny.

```python
# Create a Boolean Series for sunny days: sunny
sunny = df_clean['sky_condition']=='CLR'

# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample('D').sum()

# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample('D').count()

# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()
```

> #### Heat or humidity
> > * Dew point is a measure of relative humidity based on pressure and temperature. A dew point above 65 is considered uncomfortable while a temperature above 90 is also considered uncomfortable.
> > * explore the maximum temperature and dew point of each month. The columns of interest are 'dew_point_faren' and 'dry_bulb_faren'. 
> > * After resampling them appropriately to get the maximum temperature and dew point in each month, generate a histogram of these values as subplots. 

```python
# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='hist',  bins=8, alpha=0.5, subplots=True)

# Show the plot
plt.show()

```

> #### Probability of high temperatures
> > * will use a CDF plot to determine the probability of the 2011 daily maximum temperature in August being above the 2010 climate normal value.

```python
# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-08','Temperature'].max()
print(august_max)

# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011
august_2011 = df_clean.loc['2011-08','dry_bulb_faren'].resample('D').max()

# Filter out days in august_2011 where the value exceeded august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011 > august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)

# Display the plot
plt.show()
```