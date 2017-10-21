# Manipulating DataFrames with pandas

## Extracting and transforming data
### Indexing DataFrames
> #### Positional(iloc) and labeled(loc) indexing

```python
# Assign the row position of election.loc['Bedford']: x
x = 4

# Assign the column position of election.loc['Bedford']: y
y = 4

# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])

```
> #### Indexing and column rearrangement
> There are circumstances in which it's useful to modify the order of your DataFrame columns


```python
# Import pandas
import pandas as pd

# Read in filename and set the index: election
election = pd.read_csv(filename, index_col='county')

# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]

# Print the output of results.head()
print(results.head())

```

### Slicing DataFrames
> #### Slicing rows

```python
# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election['Perry':'Potter']

# Print the p_counties DataFrame
print(p_counties)

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election['Potter':'Perry':-1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

```

> #### Slicing columns

```python
# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']

# Print the output of left_columns.head()
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']

# Print the output of middle_columns.head()
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]

# Print the output of right_columns.head()
print(right_columns.head())

```

> #### Subselecting DataFrames with lists
> use lists to select specific row and column labels with the .loc[] accessor

```python
# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)

```


### Filtering DataFrames
> #### Thresholding data

```python
# Create the boolean array: high_turnout
high_turnout = election['turnout'] > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)

```

> #### Filtering columns using other columns
> > * use boolean selection to filter the rows where the margin was less than 1. 
> > * then convert these rows of the 'winner' column to np.nan to indicate that these results are too close to declare a winner.

```python
# Import numpy
import numpy as np

# Create the boolean array: too_close
too_close = election['margin'] < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.loc[too_close, 'winner'] = np.nan

# Print the output of election.info()
print(election.info())

```

> #### Filtering using NaNs
> In certain scenarios, it may be necessary to remove rows and columns with missing data from a DataFrame. 
> > * .dropna() method is used to perform this action.
> > * thresh= keyword argument to drop columns from the full dataset that have more than 1000 missing values.

```python
# Select the 'age' and 'cabin' columns: df
df = titanic[['age','cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
# 加入any參數的意思：行中只要有一個為nan就drop掉
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
# 加入all參數的意思：行全為nan才會drop掉
print(df.dropna(how='all').shape)


# Call .dropna() with thresh=1000 and axis='columns' and print the output of .info() from titanic
print(titanic.dropna(thresh=1000, axis='columns').info())

```

### Transforming DataFrames
> #### Using apply() to transform a column
> .apply() method can be used on a pandas DataFrame to apply an arbitrary Python function to every element

```python
# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())
```

> #### Using .map() with a dictionary
> .map() method is used to transform values according to a Python dictionary look-up

```python
# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# Print the output of election.head()
print(election.head())

```

> #### Using vectorized functions
> > * When performance is paramount, you should avoid using .apply() and .map() because those constructs perform Python for-loops over the data stored in a pandas Series or DataFrame
> > * By using vectorized functions instead, you can loop over the data at the same speed as compiled code (C, Fortran, etc.)! NumPy, SciPy and pandas come with a variety of vectorized functions (called Universal Functions or UFuncs in NumPy).
> > * zscore UFunc will take a pandas Series as input and return a NumPy array
> > * assign the values of the NumPy array to a new column in the DataFrame
	

```python
# Import zscore from scipy.stats
from scipy.stats import zscore 

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())

```

## Advanced indexing
### Index objects and labeled data

> #### Changing index of a DataFrame
> > * A list comprehension is a succinct way to generate a list in one line. For example, the following list comprehension generates a list that contains the cubes of all numbers from 0 to 9: cubes = [i**3 for i in range(10)]. 

```python
# Create the list of new indexes: new_idx
new_idx = [sales.upper() for sales in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)
```
> #### Changing index name labels

```python
# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCTS'

# Print the sales dataframe again
print(sales)
```
> #### Building an index, then a DataFrame
> > * build the DataFrame and index independently, and then put them together. If you take this route, be careful, as any mistakes in generating the DataFrame or the index can cause the data and the index to be aligned incorrectly.

```python
# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months

# Print the modified sales DataFrame
print(sales)
```

### Hierarchical indexing
> #### Extracting data with a MultiIndex
> Extracting elements from the outermost level of a MultiIndex is just like in the case of a single-level Index. You can use the .loc[] 

```python
# Print sales.loc[['CA', 'TX']]
print(sales.loc[['CA', 'TX']])

# Print sales['CA':'TX']
print(sales['CA':'TX'])

```

> #### Setting & sorting a MultiIndex

```python
# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])

# Sort the MultiIndex: sales
sales = sales.sort_index()

# Print the sales DataFrame
print(sales)
```

> #### Using .loc[] with nonunique indexes
> >  it is always preferable to have a meaningful index that uniquely identifies each row. Even though pandas does not require unique index values in DataFrames

```python
# Set the index to the column 'state': sales
sales = sales.set_index('state')

# Print the sales DataFrame
print(sales)

# Access the data from 'NY'
print(sales.loc['NY'])

```
> #### Indexing multiple levels of a MultiIndex

```python
# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[('NY', 1)]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA', 'TX'], 2), :]

# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None), 2), :]

```

## Rearranging and reshaping data
### Pivoting DataFrames
> #### Pivoting a single variable

```python
# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index='weekday', columns='city', values='visitors')

# Print the pivoted DataFrame
print(visitors_pivot)
```
![1](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/1.png)

> #### Pivoting all variables

```python
# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index='weekday', columns='city', values='signups')

# Print signups_pivot
print(signups_pivot)

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index='weekday', columns='city')

# Print the pivoted DataFrame
print(pivot)
```
![2](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/2.png)


#### Stacking & unstacking DataFrames
> #### Stacking & unstacking

```python
# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level='weekday')

# Print the byweekday DataFrame
print(byweekday)

# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level='weekday'))
```
![3](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/3.png)

> #### Restoring the index order
> > *  use .swaplevel(0, 1) to flip the index levels.
> > * To sort them, you will have to follow up with a .sort_index()

```python
# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))
```

![4](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/4.png)


#### Melting DataFrames
> #### Adding names for readability
> > * You can explicitly specify the columns that should remain in the reshaped DataFrame with id_vars, and list which columns to convert into values with value_vars
> > * if you don't pass a name to the values in pd.melt(), you will lose the name of your variable. You can fix this by using the value_name keyword argument

```python
# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index() 

# Print visitors_by_city_weekday
print(visitors_by_city_weekday)

# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')

# Print visitors
print(visitors)

```
![5](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/5.png)

> #### Going from wide to long
> > You can move multiple columns into a single column (making the data long and skinny) by "melting" multiple columns.

```python
# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday', 'city'])

# Print skinny
print(skinny)

```
![6](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/6.png)


### Pivot tables
> #### Setting up a pivot table
> > a pivot table allows you to see all of your variables as a function of two other variables.

```python
# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index='weekday',columns='city')

# Print by_city_day
print(by_city_day)
```
![7](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/7.png)

> #### Using other aggregations in pivot tables
> > use aggregation functions with in a pivot table by specifying the aggfunc parameter.

```python
# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday',aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday',aggfunc=len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))
```
![8](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/8.png)

> #### Using margins in pivot tables
> > Sometimes it's useful to add totals in the margins of a pivot table

```python
# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index='weekday', aggfunc=sum)

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total 
signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum, margins=True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)
```

![9](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/9.png)

## Grouping data
### Categoricals and groupby
> #### Grouping by multiple columns

```python
# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked','pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

# Print count_mult
print(count_mult)
```
![10](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/10.png)

> #### Grouping by another series

```python
# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())
```

### Groupby and aggregation
> #### Computing multiple aggregates of multiple columns
> > The .agg() method can be used with a tuple or list of aggregations as input

```python
# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max','median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare','median')])

```
![11](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/11.png)

> Aggregating on index levels/fields

```pythob
# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv', index_col=['Year','region','Country']).sort()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level = ['Year','region' ])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))
```

![12](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/12.png)


### Groupby and transformation
> #### Detecting outliers with Z-Scores
> > * The z-score is also useful to find outliers: a z-score value of +/- 3 is generally considered to be an outlier.

```python
# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

```
> #### Filling missing data (imputation) by group
> > * using the .dropna() method to drop missing values
> > * transform each group with a custom function to call .fillna() and impute the median value.

```python
# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))
```

> #### Other transformations with .apply
> > The .apply() method when used on a groupby object performs an arbitrary function on each of the groups. These functions can be aggregations, transformations or more complex workflows. The .apply() method will then combine the results in an intelligent way.


```python
def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})
 
 =====================================================================
 =====================================================================
 
# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])


```

### Groupby and filtering
> #### Grouping and filtering with .apply()

```python
def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)

    return gr.loc[c_passengers, 'survived'].mean()
     
 =====================================================================
 =====================================================================
 
 # Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival and print the result
c_surv_by_sex =  by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)

```

> #### Grouping and filtering with .filter()
> > with the .filter() method to remove whole groups of rows from a DataFrame based on a boolean condition.

```python
# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)
```

> #### Filtering and grouping with .map()

```python
# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])['survived'].mean()
print(survived_mean_2)

```

![13](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/13.png)


## Bringing it all together
### Case Study - Summer Olympics
> #### Grouping and aggregating
> > Suppose you have loaded the data into a DataFrame medals. You now want to find the total number of medals awarded to the USA per edition. To do this, filter the 'USA' rows and use the groupby() function to put the 'Edition' column on the index:

```python
USA_edition_grouped = medals.loc[medals.NOC == 'USA'].groupby('Edition')
USA_edition_grouped['Medal'].count()
```

> #### Using .value_counts() for ranking
> > * use the pandas Series method .value_counts() to determine the top 15 countries ranked by total number of medals.
> > * .value_counts() sorts by values by default. The result is returned as a Series of counts indexed by unique entries from the original Series with values (counts) ranked in descending order.

```python
# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))
```

> #### Using .pivot_table() to count medals by type
> > * use a pivot table to compute how many separate bronze, silver and gold medals each country won
> > * use .pivot_table() first to aggregate the total medals by type.
> > * use .sum() along the columns of the pivot table to produce a new column

```python
# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC', values='Athlete',columns='Medal',aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals',ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))

```
![14](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/14.png)

### Understanding the column labels
> #### Applying .drop_duplicates()
> * What could be the difference between the 'Event_gender' and 'Gender' columns? 
> * evaluate your guess by looking at the unique values of the pairs (Event_gender, Gender) in the data. 
> * In particular, you should not see something like (Event_gender='M', Gender='Women'). 
> * However, you will see that, strangely enough, there is an observation with (Event_gender='W', Gender='Men').

```python
# Select columns: ev_gen
ev_gen = medals[['Event_gender','Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)

```

![15](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/15.png)

> #### Finding possible errors with .groupby()
> > * use .groupby() to continue your exploration. Your job is to group by 'Event_gender' and 'Gender' and count the rows.
> > * see that there is only one suspicious row: This is likely a data error.

```python
# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender','Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)

```

![16](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/16.png)

> #### Locating suspicious data
> > now inspect the suspect record by locating the offending row

```python
# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)

```
![17](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/17.png)


### Constructing alternative country rankings
> #### Using .nunique() to rank by distinct sports
> > * want to know which countries won medals in the most distinct sports. 
> > * The .nunique() method is the principal aggregation here. Given a categorical Series S, S.nunique() returns the number of distinct categories.

```python
# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 5 rows of Nsports
print(Nsports.head(5))
```
![18](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/18.png)

> #### Counting USA vs. USSR Cold War Olympic Sports
> > aggregate the number of distinct sports in which the USA and the USSR won medals during the Cold War years.

```python
# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition'] <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)
```
![19](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/19.png)

> #### Counting USA vs. USSR Cold War Olympic Medals
> > * see which country, the USA or the USSR, won the most medals consistently over the Cold War period.
> > * a pivot table with years ('Edition') on the index and countries ('NOC') on the columns. The entries will be the total number of medals each country won that year. If the country won no medals in a given edition, expect a NaN in that entry of the pivot table.
> > * to slice the Cold War period and subset the 'USA' and 'URS' columns.
> > * make a Series from this slice of the pivot table that tells which country won the most medals in that edition using .idxmax(axis='columns'). If .max() returns the maximum value of Series or 1D array, .idxmax() returns the index of the maximizing element. The argument axis=columns or axis=1 is required because, by default, this aggregation would be done along columns for a DataFrame.
> > * The final Series contains either 'USA' or 'URS' according to which country won the most medals in each Olympic edition. You can use .value_counts() to count the number of occurrences of each

```python
# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition',columns='NOC',values='Athlete',aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_usr_medals
cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals 
most_medals = cold_war_usa_usr_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())

```
![20](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/20.png)


### Reshaping DataFrames for visualization
> #### Visualizing USA Medal Counts by Edition: Line Plot

```python
# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()
```
![21](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/21.png)


> #### Visualizing USA Medal Counts by Edition: Area Plot

```python
# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()
```
![22](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/22.png)

> #### Visualizing USA Medal Counts by Edition: Area Plot with Ordered Medal
> > noticed that the medals are ordered according to a lexicographic (dictionary) ordering: Bronze < Gold < Silver. However, you would prefer an ordering consistent with the Olympic rules: Bronze < Silver < Gold.

```python
# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values = medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()
```
![23](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/23.png)




