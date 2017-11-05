# Python Data Science Toolbox (Part 2)
## 1. Using iterators in PythonLand

### Iterators vs. iterables

> > * Iterable
> > 	* lists, strings, dictionaries, file connections
> > 	* An object with an associated iter() method
> > 	* Applying iter() to an iterable creates an iterator
> > * Iterator
> > 	* Produces next value with next() 

```python
## Iterating over iterables (1)

# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash :
    print(person)


# Create an iterator for flash: superspeed
superspeed = iter(flash)

# Print each item from the iterator
print(next(superspeed))
print(next(superspeed))

########################################################
########################################################

## Iterating over iterables (2)

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)
    
########################################################
########################################################

# You've been using the iter() function to get an iterator object, as well as 
# the next() function to retrieve the values one by one from the iterator object.
# There are also functions that take iterators as arguments. For example, the 
# list() and sum() functions return a list and the sum of elements, respectively.

# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)

```

### Enumerate vs. Zip

> enumerate() returns an enumerate object that produces a sequence of tuples, and each of the tuples is an index-value pair.

```python
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pride']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in mutant_list:
    print(index1, value1)

# Change the start index
for index2, value2 in list(enumerate(mutants, start=1)):
    print(index2, value2)

```

> >  * zip(), which takes any number of iterables and returns a zip object that is an iterator of tuples. If you wanted to print the values of a zip object, you can convert it into a list and then print it. Printing just a zip object will not return the values unless you unpack it firs

```pyhton
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)
    
########################################################
########################################################

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

```


### Using iterators for big data

```pyhton
# Extracting information for large amounts of Twitter data
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict
        
########################################################
# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)

```


## 2. List comprehensions and generaton

### List comprehensions

> * Collapse for loops for building lists into a single line
> * Components
	* Iterable
	* Iterator variable (represent members of iterable)
	* Output expression

```python
# Writing list comprehensions

# Create list comprehension: squares
squares = [i**2 for i in range(0,10)]

########################################################
########################################################

# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

```

### Advanced comprehensions

```python
# Using conditionals in comprehensions (1)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]

# Print the new list
print(new_fellowship)

########################################################
########################################################

# Using conditionals in comprehensions (2)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)

########################################################
########################################################

# Dict comprehensions

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member:len(member) for member in fellowship}

# Print the new list
print(new_fellowship)

```

### Introduction to generator expressions

> * Use ( ) instead of [ ]
> * List comprehension - returns a list
> * Generators - returns a generator object (用於處理大數量range(10*1000000))
> * Both can be iterated over

```python
# Create generator object: result
result = (num for num in range(0,31))

# Print the first 2 values
print(next(result))
print(next(result))

########################################################
########################################################

# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)
    
```

> > * Generator functions are functions that, like generator expressions, yield a series of values, instead of returning a single value. A generator function is defined as you do a regular function, but whenever it generates a value, it uses the keyword yield instead of return.

```python
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)
```

### Wrapping up comprehensions and generators

```pyhton
# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)
```

## 3. Bringing it all together!

### Welcome to the case study!

> * Data on world economies for over half a century
> * Indicators
>  * Population
>  * Electricity consumption
>  * CO2 emissions
>  * Literacy rates 識字率
>  * Unemployment


```python
# plot_pop() which takes two arguments: the filename of the file to be processed, and the country code of the rows you want to process in the dataset
def plot_pop(filename, country_code):
	
	 # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)
    
    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
    	 # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]
        
        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])
                    
        # Turn zip object into list: pops_list
        pops_list = list(pops)
        
        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]
        
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)
        
        # Plot urban population data
       data.plot(kind='scatter', x='Year', y='Total Urban Population')
       plt.show()
       
########################################################
########################################################

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn,'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn,'ARB')
  
```



