# Cleaning Data in Python

## 1. Exploring your data
> *  you've just got a brand new dataset and are itching to start exploring it. But where do you begin, and how can you be sure your dataset is clean?

### Diagnose data for cleaning
> #### Common data problems
> 
>  * Inconsistent column names(ex. country的拼法有英文拼法也有法文拼法..)
>  * Missing data
>  * Outliers
>  * Duplicate rows
>  * Untidy
>  * Need to process columns
>  * Column types can signal unexpected data values
>
> #### Loading and viewing your data
> 
> >  * .head() and .tail()
> >  * .shape and .columns attributes let you see the shape of the DataFrame and obtain a list of its columns

```python
# Import pandas
import pandas as pd

# Read the file into a DataFrame: df
df = pd.read_csv('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

# Print the info of df
print(df.info())
```
> #### Further diagnosis
> > * The .info() method provides important information about a DataFrame, such as the number of rows, number of columns, number of non-missing values in each column, and the data type stored in each column. 

![291](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/291.png)

### Exploratory data analysis
> #### Calculating summary statistics
> > * .describe() method to calculate summary statistics of your data

![292](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/292.png)

> #### Frequency counts for categorical data
> > * .describe() can only be used on numeric columns. 
> > * So how can you diagnose data issues when you have categorical data? 
> > * One way is by using the .value_counts() method, which returns the frequency counts for each unique value in a column!
> > * This method also has an optional parameter called dropna which is True by default. 
> > * What this means is if you have missing data in a column, it will not give a frequency count of them. You want to set the dropna column to False so if there are missing values in a column, it will give you the frequency counts.

```python
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))

# Print the value_counts for 'State'
print(df['State'].value_counts(dropna=False))

# Print the value counts for 'Site Fill'
print(df['Site Fill'].value_counts(dropna=False))

```

![293](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/293.png)

### Visual exploratory data analysis
> #### Visualizing single variables with histograms
> > * The .plot() method allows you to create a plot of each column of a DataFrame. 
> >  * The kind parameter allows you to specify the type of plot to use - kind='hist'
> > * The keyword arguments logx=True or logy=True can be passed in to .plot() depending on which axis you want to rescale.

![294](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/294.png)

> #### Visualizing multiple variables with boxplots
> > * The pandas .boxplot() method is a quick way to do this, in which you have to specify the column and by parameters.
> > * You can see the 2 extreme outliers are in the borough of Manhattan. An initial guess could be that since land in Manhattan is extremely expensive, these outliers may be valid data points. 

```python
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)

# Display the plot
plt.show()
```
![295](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/295.png)

> #### Visualizing multiple variables with scatter plots
> > * Boxplots are great when you have a numeric column that you want to compare across different categories. 
> > * When you want to visualize two numeric columns, scatter plots are ideal.
> > * In general, from the second plot it seems like there is a strong correlation between 'initial_cost' and 'total_est_fee'. 
> > * In addition, take note of the large number of points that have an 'initial_cost' of 0. It is difficult to infer any trends from the first plot because it is dominated by the outliers.

```python
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind='scatter',x='initial_cost', y='total_est_fee', rot=70)
plt.show()
```
![296](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/296.png)

![297](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/297.png)

## 2. Tidying data for analysis
### Tidy data
> #### Recognizing tidy data
> > For data to be tidy, it must have:
> >
> > * Each variable as a separate column.
> > * Each row as a separate observation.
> > * df2. is not tidy, each variable is not a separate column.

![298](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/298.png)

![299](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/299.png)

> #### Reshaping your data using melt
> 
> > * pd.melt(): There are two parameters you should be aware of: id_vars and value_vars
> > * id_vars: represent the columns of the data you do not want to melt (i.e., keep it in its current shape)
> > * value_vars: represent the columns you do wish to melt into rows. By default, if no value_vars are provided, all columns not set in the id_vars will be melted
> > * var_name: rename the variable column by specifying an argument to the var_name
> > * value_name: the value column by specifying an argument to the value_name

```python
# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=df, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading'
```

![300](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/300.png)

> #### Customizing melted data

```python
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=df, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())
```

![301](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/301.png)

### Pivoting data

> #### Pivot data
> > * Pivoting data is the opposite of melting it.
> > * While melting takes a set of columns and turns it into a single column, pivoting will create a new column for each unique value in a specified column
> > pivot_table()
> > * index: which you can use to specify the columns that you don't want pivoted: It is similar to the id_vars parameter of pd.melt() 
> > * Two other parameters that you have to specify are columns (the name of the column you want to pivot), and values (the values to be used when the column is pivoted). 

```python
# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())
```

![302](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/302.png)

> #### Resetting the index of a DataFrame
> > * reset_index(): After pivoting airquality_melt in the previous exercise, you didn't quite get back the original DataFrame.

```python
# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())
```

![303](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/303.png)

> #### Pivoting duplicate values
> > * Pivoting duplicate values, the aggfunc parameter, you can not only reshape your data, but also remove duplicates. 

```python
# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())
```

![304](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/304.png)

### Beyond melt and pivot
> #### Splitting a column with .str
> > * In this exercise, you're going to tidy the 'm014' column, which represents males aged 0-14 years of age. 
> > * In order to parse this value, you need to extract the first letter into a new column for gender, and the rest into a column for age_group
> > * Notice the new 'gender' and 'age_group' columns you created. It is vital to be able to split columns as needed so you can access the data that is relevant to your question.

```python
# Melt tb: tb_melt
tb_melt = pd.melt(frame=tb, id_vars=['country', 'year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())
```

![305](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/305.png)

![306](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/306.png)

> #### Splitting a column with .split() and .get()
> > * Notice that the data has column names such as Cases_Guinea and Deaths_Guinea. Here, the underscore _ serves as a delimiter between the first part (cases or deaths), and the second part (country).
> > * This time, you cannot directly slice the variable by position as in the previous exercise. 
> > * You now need to use Python's built-in string method called .split()
> > * using the .get() method to retrieve the 0 or 1 index, depending on the part you want.

```python
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())
```

![307](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/307.png)

![308](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/308.png)

## 3. Combining data for analysis

### Concatenating data
> #### Combining rows of data
> > pd.concat() function, but this time with the keyword argument axis=1. The default, axis=0, is for a row-wise concatenation.
 
```python
# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1,uber2,uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())
```
![309](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/309.png)

> #### Combining columns of data

```python
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis = 1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())
```
![310](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/310.png)

### Finding and concatenating data
> #### Finding files that match a pattern
> > * glob module has a function called glob that takes a pattern and returns a list of the files in the working directory that match that pattern.

```python
# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())
```
![311](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/311.png)

> #### Iterating and concatenating all matches

```python
# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())

```
![312](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/312.png)

###  Merge data
> #### 1-to-1 data merge
> > * Two DataFrames have been pre-loaded: site and visited
> > * Your task is to perform a 1-to-1 merge of these two DataFrames using the 'name' column of site and the 'site' column of visited.

```python
# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')
```
![313](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/313.png)

> #### Many-to-1 data merge
> > * Note that this time, visited has multiple entries for the site column
> > * Notice how the site data is duplicated during this many-to-1 merge!

```python
# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')
```
![314](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/314.png)

> #### Many-to-many data merge
> > * work with the site and visited DataFrames from before, and a new survey DataFrame. Your task is to merge site and visited as you did in the earlier exercises. You will then merge this merged DataFrame with survey.

```python
# Merge site and visited: m2m
m2m = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Merge m2m and survey: m2m
m2m = pd.merge(left=m2m, right=survey, left_on='ident', right_on='taken')

# Print the first 20 lines of m2m
print(m2m.head(20))

```
![315](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/315.png)

![316](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/316.png)

## 4. Cleaning data for analysis

### Data types
> #### Converting data types
> > * ensuring all categorical variables in a DataFrame are of type category reduces memory usage.
> > * astype()
> > * By converting sex and smoker to categorical variables, the memory usage of the DataFrame went down from 13.4 KB to 10.1KB. 

```python
# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())
```
![317](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/317.png)

> #### Working with numeric data
> > * you expect the data type of a column to be numeric (int or float), but instead it is of type object, this typically means that there is a non numeric value in the column, which also signifies bad data.
> > * You can use the pd.to_numeric() function to convert a column into a numeric data type.

```python
# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())

```
![318](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/318.png)

### Using regular expressions to clean strings
> #### String parsing with regular expressions
> > * Compile a pattern that matches a phone number of the format xxx-xxx-xxxx

```python
# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result)) # True

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result)) # False
```

> #### Extracting numerical values from strings
> > * When using a regular expression to extract multiple numbers (or multiple pattern matches, to be exact), you can use the re.findall() function.
> > *  your regular expression successfully extracted the numeric values 10 and 1 from the string!

```python
# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)  # ['10', '1']
```

> #### Pattern matching

```python
# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1) # True

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d{3}\.\d{2}', string='$123.45'))
print(pattern2) # True

# Write the third pattern
pattern3 = bool(re.match(pattern='\w*', string='Australia'))
print(pattern3) # True
```

### Using functions to clean data
> #### Custom functions to clean data
> .apply() method to apply a function across entire rows or columns of DataFrames. 

```python
# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == 'Male':
        return 1
    
    # Return 0 if sex_value is 'Female'    
    elif sex_value == 'Female':
        return 0
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply(recode_sex)

# Print the first five rows of tips
print(tips.head())
```
![319](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/319.png)

> #### Lambda functions
>  Instead of using the def syntax that you used in the previous exercise, lambda functions let you make simple, one-line functions.
> > * Notice how the 'total_dollar_re' and 'total_dollar_replace' columns are identical.

```python
# Write the lambda function using replace
tips['total_dollar_replace'] = tips['total_dollar'].apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips['total_dollar'].apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())
```
![320](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/320.png)

### Duplicate and missing data
> #### Dropping duplicate data
> > * drop_duplicates()
> > * After dropping duplicates, the DataFrame has gone from having 24092 entries to only 317!

```python
# Create the new DataFrame: tracks
tracks = billboard[['year','artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())

```
![321](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/321.png)

> #### Filling missing data 
> > * fillna()
> > * There are no longer any missing values in the Ozone column of this DataFrame!

```python
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Print the info of airquality
print(airquality.info())
```
![322](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/322.png)


### Testing with asserts
> >  * chain two .all() methods (that is, .all().all()). 
> > The first .all() method will return a True or False for each column, while the second .all() method will return a single True or False.

```python
# Assert that there are no missing values
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
assert (ebola >= 0).all().all()
```

## 5. Case study

### Putting it all together
> #### Exploratory analysis
> > * Whenever you obtain a new dataset, your first task should always be to do some exploratory analysis to get a better understanding of the data and diagnose it for any potential issues.
>  * use pandas methods such as .head(), .info(), and .describe(), and DataFrame attributes like .columns and .shape to explore it 

![323](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/323.png)

> #### Visualizing your data
> > * visually check the data for insights as well as errors
> > * there are a surprising number of countries that fall on the diagonal line. In fact, examining the DataFrame reveals that the life expectancy for 140 of the 260 countries did not change at all in the 19th century! This is possibly a result of not having access to the data for all the years back then. 

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create the scatter plot
g1800s.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()

```

![324](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/324.png)

> #### Thinking about the question at hand
> > Before continuing, however, it's important to make sure that the following assumptions about the data are true:

> > * 'Life expectancy' is the first column (index 0) of the DataFrame.
> > * The other columns contain either null or numeric values.
> > * The numeric values are all greater than or equal to 0.
> > * There is only one instance of each country. 

```python
def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()[1:-1]
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1

```

> #### Assembling your data
> > * three DataFrames have been pre-loaded: g1800s, g1900s, and g2000s. These contain the Gapminder life expectancy data for, respectively, the 19th century, the 20th century, and the 21st century. 
> > * Your task in this exercise is to concatenate them into a single DataFrame called gapminder

```python
# Concatenate the DataFrames row-wise
gapminder = pd.concat([g1800s, g1900s, g2000s])

# Print the shape of gapminder
print(gapminder.shape)

# Print the head of gapminder
print(gapminder.head())
```

### Initial impressions of the data
> #### Reshaping your data
> > * gapminder DataFrame has a separate column for each year.
> > * What you want instead is a single column that contains the year, and a single column that represents > the average life expectancy for each year and country

```python
# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(frame=gapminder, id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())
```

![325](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/325.png)

![326](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/326.png)
 
> #### Checking the data types
> > * Now that your data is in the proper shape, you need to ensure that the columns are of the proper data type.
> > * That is, you need to ensure that country is of type object, year is of type int64, and life_expectancy is of type float64

```python
# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder.year)

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64

```

> #### Looking at country spellings
> > * your next task in the data cleaning process is to look at the 'country' column to see if there are any special or invalid characters you may need to deal with.
> > It is reasonable to assume that country names will contain:
> > * The set of lower and upper case letters.
> > * Whitespace between words.
> > * Periods for any abbreviations.
> > * not all these country names are actually invalid so maybe the assumptions need to be tweaked a little. However, there certainly are a few cases worth further investigation, such as St. Barth?lemy. 
> > * Whenever you are dealing with columns of raw data consisting of strings, it is important to check them for consistency like this 

```python
# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
# 1. Anchor the pattern to match exactly what you want by placing a ^ in the beginning and $ in the end
# 2. Use A-Za-z to match the set of lower and upper case letters
# 3. Use \. to match periods, and \s to match whitespace between words
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)
```
![327](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/327.png)

> #### More data cleaning and processing
> > * dropna() method has the default keyword arguments axis=0 and how='any', which specify that rows with any missing values should be dropped.
> > * the number of rows in the DataFrame has gone down from 169260 to 43857. In general, you should avoid dropping too much of your data, but if there is no reasonable way to fill in or impute missing values, then dropping the missing data may be the best solution.

```python
# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna(how='any')

# Print the shape of gapminder
print(gapminder.shape)  # (43857, 3)

```

> #### Wrapping up
> > * Now that you have a clean and tidy dataset, you can do a bit of visualization and aggregation. 
> > * In this exercise, you'll begin by creating a histogram of the life_expectancy column. You should not get any values under 0 and you should see something reasonable on the higher end of the life_expectancy age range

```python
# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')
```
![328](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/328.png)

![329](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/329.png)