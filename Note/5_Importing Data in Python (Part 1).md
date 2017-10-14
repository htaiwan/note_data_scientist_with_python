# Importing Data in Python (Part 1)
## 1. Introduction and flat files

**Importing entire text files vs. Importing text files line by line**

```python
# Open a file: file
file = open('moby_dick.txt', mode = 'r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)

########################################################
########################################################

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

```

**What is Flat files**

> * Text files containing records
> * That is, table data
> * Record: row of fields or a ributes
> * Column: feature or a ribute
> * How do you import flat files - Two main packages: NumPy, pandas

**Using NumPy to import flat files**

> * **delimiter** changes the delimiter that loadtxt() is expecting, for example, you can use ',' and '\t' for comma-delimited and tab-delimited respectively; 
> 
> * **skiprows** allows you to specify how many rows (not indices) you wish to skip;
>  
> * **usecols** takes a list of the indices of the columns you wish to keep.
>
> * Due to the header, if you tried to import it as-is using np.loadtxt(), Python would throw you a ValueError and tell you that it could not convert string to float. There are two ways to deal with this: firstly, you can set the data type argument **dtype** equal to str (for string).
> 
> * Much of the time you will need to import datasets which have different datatypes in different columns; one column may contain strings and another floats, for example. **The function np.loadtxt() will freak at this. There is another function, np.genfromtxt(), which can handle such structures.**


```python
# Import package
import numpy as np

# Assign filename: file
file = 'seaslug.txt'

# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()

########################################################
########################################################

data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)

########################################################
########################################################

# np.recfromcsv() that behaves similarly to np.genfromtxt(), except that its default dtype is Non

# Assign the filename: file
file = 'titanic.csv'

# Import file using np.recfromcsv: d
d = np.recfromcsv(file, delimiter=',', names=True)

```

**Using pandas to import flat files as DataFrames**

> The pandas package is also great at dealing with many of the issues you will encounter when importing data as a data scientist, such as **comments** occurring in flat files, **empty lines and missing values**. Note that missing values are also commonly referred to as NA or NaN
> 
> * **na_values** takes a list of strings to recognize as NA/NaN, in this case the string 'Nothing'.
> * **comment** takes characters that comments occur after in the file, which in this case is '#' 
> * **sep** (the pandas version of delim)

```python
# Import pandas as pd
import pandas as pd

# Assign the filename: file
file = 'titanic.csv'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')

# View the head of the DataFrame
print(df.head())

```

## 2. Importing data from other file types

**Pickled files**
> * File type native to Python
* Motivation: many datatypes for which it isn’t obvious how to store them
* Pickled files are serialized
* Serialize = convert object to bytestream

```python
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))
```
**Importing sheets from Excel files**

```python
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'battledeath.xlsx'

# Load spreadsheet: xl
xl = pd.ExcelFile(file)

# Print sheet names
print(xl.sheet_names)

# Parse the first sheet and rename the columns: df1
df1 = xl.parse(0, skiprows=1, names=['Country','AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xl.parse(1, parse_cols=[0], skiprows=1, names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())

```

**Importing SAS/Stata files using pandas**

```python
# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())


# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()

########################################################
########################################################

# Import pandas
import pandas as pd

# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of coutries')
plt.show()

```
**Importing HDF5 files**

```python
# Import packages
import numpy as np
import h5py

# Assign filename: file
file = 'LIGO_data.hdf5'

# Load file: data
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)
```
**Importing MATLAB files**

```python
# Import package
import scipy.io

# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')

# Print the datatype type of mat
print(type(mat))
```

## 3. Working with relational databases

**Creating a database engine in Python**

```python
# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)

```

**Querying relational databases in Python**

```python
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
with engine.connect() as con:
    rs = con.execute("SELECT * from Employee ORDER BY BirthDate")
  # rs = con.execute("SELECT * FROM Employee WHERE EmployeeId >= 6")
    df = pd.DataFrame(rs.fetchall())

    # Set the DataFrame's column names
    df.columns = rs.keys()

# Print head of DataFrame
print(df.head())

```

**Querying relational databases directly with pandas**

```python
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate", engine)

# Print head of DataFrame
print(df.head())

```

**Advanced Querying: exploiting table relationships**

> INNER JOIN in Python 

```python
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())

########################################################
########################################################

# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000", engine)

# Print head of DataFrame
print(df.head())

```