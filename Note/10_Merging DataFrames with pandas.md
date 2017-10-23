# Merging DataFrames with pandas

## Preparing data
### Reading multiple data files
> #### Reading DataFrames from multiple files in a loop

```python
# Import pandas
import pandas as pd

# Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']

# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))
``` 

> #### Combining DataFrames from multiple data files
> combine the three DataFrames from earlier exercises - gold, silver, & bronze - into a single DataFrame called medals

```python
# Import pandas
import pandas as pd

# Make a copy of gold: medals
medals = gold.copy()

# Create list of new column labels: new_labels
new_labels = ['NOC', 'Country', 'Gold']

# Rename the columns of medals using new_labels
medals.columns = new_labels

# Add columns 'Silver' & 'Bronze' to medals
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']

# Print the head of medals
print(medals.head())

``` 

### Reindexing DataFrames
> #### Sorting DataFrame with the Index & columns
> the principal methods for doing this are .sort_index() and .sort_values()

```python
# Import pandas
import pandas as pd

# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col='Month')

# Print the head of weather1
print(weather1.head())

# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()

# Print the head of weather2
print(weather2.head())

# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending=False)

# Print the head of weather3
print(weather3.head())

# Sort weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values('Max TemperatureF')

# Print the head of weather4
print(weather4.head())
``` 

> #### Reindexing DataFrame from a list
> Sorting methods are not the only way to change DataFrame Indexes. There is also the .reindex() method
> > * use a list of all twelve month abbreviations and subsequently apply the .ffill() method to forward-fill the null entries when upsampling

```python
# Import pandas
import pandas as pd

# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)

# Print weather2
print(weather2)

# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()

# Print weather3
print(weather3)

``` 
> #### Reindexing using another DataFrame Index
> Another common technique is to reindex a DataFrame using the Index of another DataFrame

```python
# Import pandas
import pandas as pd

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)
```
![24](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/24.png)


### Arithmetic with Series & DataFrames
> #### Adding unaligned DataFrames
> If you were to add these two DataFrames by executing the command total = january + february

![25](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/25.png)

> #### Broadcasting in arithmetic formulas
> ordinary arithmetic operators (like +, -, *, and /) broadcast scalar values to conforming DataFrames when combining scalars & DataFrames in arithmetic expressions.

```python
# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF','Mean TemperatureF','Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')

# Print first 5 rows of temps_c
print(temps_c.head())
```

> #### Computing percentage growth of GDP
> > * GDP.csv, which contains quarterly data
> > * resample it to annual sampling and then compute the annual growth of GDP

```python
import pandas as pd

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv',parse_dates=True, index_col='DATE')

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp['2008':]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

# Print yearly again
print(yearly)
```
![26](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/26.png)

> #### Converting currency of stocks
> > * Using the daily exchange rate to Pounds Sterling

```python
# Import pandas
import pandas as pd

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv',parse_dates=True, index_col='Date')

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv',parse_dates=True, index_col='Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500.loc[:,['Open','Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')

# Print the head of pounds
print(pounds.head())

``` 
![27](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/27.png)

 
## Concatenating data
### Appending & concatenating Series
> #### Appending Series with nonunique Indices
> the command combined = bronze.append(silver)

![28](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/28.png)

> #### Appending pandas Series

```python
# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', parse_dates=True, index_col='Date')

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', parse_dates=True, index_col='Date')

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar  = pd.read_csv('sales-mar-2015.csv', parse_dates=True, index_col='Date')

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())
``` 

> #### Concatenating pandas Series along row axis
> > * pd.concat() with a list of Series to achieve the same result that you would get by chaining calls to .append()
> > * between pd.concat() and pandas' .append() method. One way to think of the difference is that .append() is a specific case of a concatenation, while pd.concat() gives you more flexibility,

```python
# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])

# Concatenate the list: quarter1
quarter1 = pd.concat(units,axis='rows')

# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])
``` 

### Appending & concatenating DataFrames
> #### Appending DataFrames with ignore_index
> > *  specify ignore_index=True so that the index values are not used along the concatenation axis. The resulting axis will instead be labeled 0, 1, ..., n-1

```python
# Add 'year' column to names_1881 and names_1981
names_1881['year'] = 1881
names_1981['year'] = 1981

# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981,ignore_index=True)

# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)

# Print all rows that contain the name 'Morgan'
print(combined_names.loc[combined_names['name']=='Morgan'])
``` 
![29](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/29.png)

> #### Concatenating pandas DataFrames along column axis
> > * specify the keyword argument axis=1 or axis='columns'

```python
# Concatenate weather_max and weather_mean horizontally: weather
weather = pd.concat([weather_max,weather_mean], axis=1)

# Print weather
print(weather)

```

> #### Reading multiple files to build a DataFrame

```python
for medal in medal_types:

    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal
    
    # Create list of column names: columns
    columns = ['Country', medal]
    
    # Read file_name into a DataFrame: df
    medal_df = pd.read_csv(file_name,header=0,index_col='Country', names=columns)

    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals horizontally: medals
medals = pd.concat(medals, axis='columns')

# Print medals
print(medals)

``` 

### Concatenation, keys, & MultiIndexes
> #### Concatenating vertically to get MultiIndexed rows
> > * When stacking a sequence of DataFrames vertically, it is sometimes desirable to construct a MultiIndex to indicate the DataFrame from which each row originated. This can be done by specifying the keys parameter in the call to pd.concat(), which generates a hierarchical index with the labels from keys as the outermost index label

```python
for medal in medal_types:

    file_name = "%s_top5.csv" % medal
    
    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, index_col='Country')
    
    # Append medal_df to medals
    medals.append(medal_df)
    
# Concatenate medals: medals
medals = pd.concat(medals,keys=['bronze', 'silver', 'gold'], axis=0)

# Print medals in entirety
print(medals)
``` 
![30](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/30.png)

> #### Slicing MultiIndexed DataFrames
> > * sort the DataFrame and to use the pd.IndexSlice to extract specific slices

```python
# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)

# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:,'United Kingdom'],:])
```
![31](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/31.png)

> #### Concatenating horizontally to get MultiIndexed columns
> > * concatenate the DataFrames horizontally and to create a MultiIndex on the columns. 
> > * summarize the resulting DataFrame and slice some information from it.

```python
# Concatenate dataframes: february
february = pd.concat(dataframes,keys=['Hardware', 'Software', 'Service'], axis=1)

# Print february.info()
print(february.info())

# Assign pd.IndexSlice: idx
idx = pd.IndexSlice

# Create the slice: slice_2_8
slice_2_8 = february.loc['Feb. 2, 2015':'Feb. 8, 2015', idx[:, 'Company']]


# Print slice_2_8
print(slice_2_8)
``` 
![32](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/32.png)

> #### Concatenating DataFrames from a dict
> > * Three DataFrames jan, feb, and mar have been pre-loaded
> > *  by constructing a dictionary of these DataFrames and then concatenating them.

```python
# Make the list of tuples: month_list
month_list = [('january', jan), ('february', feb), ('march', mar)]

# Create an empty dictionary: month_dict
month_dict = {}

for month_name, month_data in month_list:

    # Group month_data: month_dict[month_name]
    month_dict[month_name] = month_data.groupby('Company').sum()
    
# Concatenate data in month_dict: sales
sales = pd.concat(month_dict)

# Print sales
print(sales)

# Print all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])
``` 
![33](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/33.png)

### Outer & inner joins
> #### Concatenating DataFrames with inner join
> > * dropped as part of the join since they are not present in each of bronze, silver, and gold(取交集)

```python
# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]

# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, keys= ['bronze', 'silver', 'gold'], axis=1, join='inner')

# Print medals
print(medals)
``` 
![34](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/34.png)

> #### Resampling & concatenating DataFrames with inner join

```python
# Resample and tidy china: china_annual
china_annual = china.resample('A').pct_change(10).dropna()

# Resample and tidy us: us_annual
us_annual = us.resample('A').pct_change(10).dropna()

# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual],join='inner', axis=1)

# Resample gdp and print
print(gdp.resample('10A').last())
```
![35](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/35.png)

## Merging data
### Merging DataFrames
> #### Merging on a specific column

![36](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/36.png)

> #### Merging on columns with non-matching labels
> > * specify the left_on and right_on parameters in the call to pd.merge()

```python
# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(revenue, managers, left_on='city', right_on='branch')

# Print combined
print(combined)
``` 

![37](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/37.png)

> #### Merging on multiple columns

```python
# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX','CO','IL','CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX','CO','CA','MO']

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on=['branch_id', 'city', 'state'])

# Print combined
print(combined)
```
![38](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/38.png)

### Joining DataFrames
> #### Joining by Index
> > * join the DataFrames on their indexes and return 5 rows with index labels [10, 20, 30, 31, 47] 

![39](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/39.png)

> #### Left & right merging on multiple columns
> > * By merging sales and managers with a left merge, you can identify the missing manager. Here, the columns to merge on have conflicting labels, so you must specify left_on and right_on

```python
# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue,sales,on=['city', 'state'], how='right')

# Print revenue_and_sales
print(revenue_and_sales)

# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales,managers, left_on=['city', 'state'], right_on=['branch', 'state'], how='left')

# Print sales_and_managers
print(sales_and_managers)
``` 

![40](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/40.png)

> #### Merging DataFrames with outer join

```python
# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers, revenue_and_sales)

# Print merge_default
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers, revenue_and_sales, how='outer')

# Print merge_outer
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers, revenue_and_sales,on=['city','state'], how='outer')

# Print merge_outer_on
print(merge_outer_on)
```
![41](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/41.png)

### Ordered merges
> #### Using merge_ordered()
> > * Weather conditions were recorded on separate days and you need to merge these two DataFrames together such that the dates are ordered. To do this, you'll use pd.merge_ordered() 

```python
# Perform the first ordered merge: tx_weather
tx_weather = pd.merge_ordered(austin, houston)

# Print tx_weather
print(tx_weather)

# Perform the second ordered merge: tx_weather_suff
tx_weather_suff = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus','_hus'])

# Print tx_weather_suff
print(tx_weather_suff)

# Perform the third ordered merge: tx_weather_ffill
tx_weather_ffill = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus','_hus'], fill_method='ffill')

# Print tx_weather_ffill
print(tx_weather_ffill)
``` 
![42](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/42.png)

## 綜合比較

![43](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/43.png)


## Case Study - Summer Olympics
### Medals in the Summer Olympics
> #### Loading Olympic edition DataFrame
> > * prepare a DataFrame editions from a tab-separated values (TSV) file
> > * editions has 26 rows (one for each Olympic edition, i.e., a year in which the Olympics was held) and 7 columns: 'Edition', 'Bronze', 'Gold', 'Silver', 'Grand Total', 'City', and 'Country'
> > *  won't need the overall medal counts, so you want to keep only the useful columns from editions: 'Edition', 'Grand Total', City, and Country

```python
#Import pandas
import pandas as pd

# Create file path: file_path
file_path = 'Summer Olympic medallists 1896 to 2008 - EDITIONS.tsv'

# Load DataFrame from file_path: editions
editions = pd.read_csv(file_path,sep='\t')

# Extract the relevant columns: editions
editions = editions[['Edition','Grand Total', 'City','Country']]

# Print editions DataFrame
print(editions)
``` 
![44](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/44.png)

> #### Loading IOC codes DataFrame
> > * ioc_codes from a comma-separated values (CSV) file.
> > * ioc_codes has 200 rows (one for each country) and 3 columns: 'Country', 'NOC', & 'ISO code'.
> > * only the useful columns from ioc_codes: 'Country' and 'NOC' (the column 'NOC' contains three-letter codes representing each country).

```python
# Import pandas
import pandas as pd

# Create the file path: file_path
file_path = 'Summer Olympic medallists 1896 to 2008 - IOC COUNTRY CODES.csv'

# Load DataFrame from file_path: ioc_codes
ioc_codes = pd.read_csv(file_path)

# Extract the relevant columns: ioc_codes
ioc_codes = ioc_codes[['Country', 'NOC']]

# Print first and last 5 rows of ioc_codes
print(ioc_codes.head())
print(ioc_codes.tail())
```
![45](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/45.png)

> #### Building medals DataFrame
> > * build up a dictionary medals_dict with the Olympic editions (years) as keys and DataFrames as values.
> > * Once the dictionary of DataFrames is built up, you will combine the DataFrames using pd.concat()

```python
# Import pandas
import pandas as pd

# Create empty dictionary: medals_dict
medals_dict = {}

for year in editions['Edition']:

    # Create the file path: file_path
    file_path = 'summer_{:d}.csv'.format(year)
    
    # Load file_path into a DataFrame: medals_dict[year]
    medals_dict[year] = pd.read_csv(file_path)
    
    # Extract relevant columns: medals_dict[year]
    medals_dict[year] = medals_dict[year][['Athlete','NOC','Medal']]

    # Assign year to column 'Edition' of medals_dict
    medals_dict[year]['Edition'] = year

    
# Concatenate medals_dict: medals
medals = pd.concat(medals_dict, ignore_index=True)

# Print first and last 5 rows of medals
print(medals.head())
print(medals.tail())

``` 
![46](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/46.png)


### Quantifying Performance
> #### Counting medals by country/edition in a pivot table

```python
# Construct the pivot_table: medal_counts
medal_counts = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')

# Print the first & last 5 rows of medal_counts
print(medal_counts.head())
print(medal_counts.tail())
``` 

![47](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/47.png)

> #### Computing fraction of medals per Olympic edition

```python
# Set Index of editions: totals
totals = editions.set_index('Edition')

# Reassign totals['Grand Total']: totals
totals = totals['Grand Total']

# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals, axis='rows')

# Print first & last 5 rows of fractions
print(fractions.head())
print(fractions.tail())
```
![48](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/48.png)

> #### Computing percentage change in fraction of medals won
> > * To see if there is a host country advantage, you first want to see how the fraction of medals won changes from edition to edition.

```python
# Apply the expanding mean: mean_fractions
mean_fractions = fractions.expanding().mean()

# Compute the percentage change: fractions_change
fractions_change = mean_fractions.pct_change() * 100

# Reset the index of fractions_change: fractions_change
fractions_change = fractions_change.reset_index()

# Print first & last 5 rows of fractions_change
print(fractions_change.head())
print(fractions_change.tail())
```
![49](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/49.png)

### Reshaping and plotting
> #### Building hosts DataFrame
> > *  prepare a DataFrame hosts by left joining editions and ioc_codes

```python
# Import pandas
import pandas as pd

# Left join editions and ioc_codes: hosts
hosts = pd.merge(editions,ioc_codes,how='left')

# Extract relevant columns and set index: hosts
hosts = hosts[['Edition','NOC']].set_index('Edition')

# Fix missing 'NOC' values of hosts
print(hosts.loc[hosts.NOC.isnull()])
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset Index of hosts: hosts
hosts = hosts.reset_index()

# Print hosts
print(hosts)
``` 
![50](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/50.png)

> #### Reshaping for analysis
> > * fractions_change and hosts already loaded
> > * fractions_change is a wide DataFrame of 26 rows (one for each Olympic edition) and 139 columns (one for the edition and 138 for the competing countries).

```python
# Import pandas
import pandas as pd

# Reshape fractions_change: reshaped
reshaped = pd.melt(fractions_change, id_vars='Edition', value_name='Change')

# Print reshaped.shape and fractions_change.shape
print(reshaped.shape, fractions_change.shape)

# Extract rows from reshaped where 'NOC' == 'CHN': chn
chn = reshaped.loc[reshaped['NOC']=='CHN']

# Print last 5 rows of chn with .tail()
print(chn.tail())

``` 
![51](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/51.png)

> #### Merging to compute influence
> > * Your task is to merge the two DataFrames(reshaped and hosts) and tidy the result

```python
# Import pandas
import pandas as pd

# Merge reshaped and hosts: merged
merged = pd.merge(reshaped, hosts, how='inner')

# Print first 5 rows of merged
print(merged.head())

# Set Index of merged and sort it: influence
influence = merged.set_index('Edition').sort_index()

# Print first 5 rows of influence
print(influence.head())

``` 
![52](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/52.png)

> #### Plotting influence of host country
> > * starts off with the DataFrames influence and editions

```python
# Import pyplot
import matplotlib.pyplot as plt

# Extract influence['Change']: change
change = influence['Change']

# Make bar plot of change: ax
ax = change.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])

# Display the plot
plt.show()

``` 
![53](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/53.png)