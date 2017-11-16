# Intermediate Python for Data Science

## 1. Matplotlib
### Basic plots with matplotlib
> #### Line plot

```python
# Print the last item from year and pop
print(year[-1])
print(pop[-1])


# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)

# Display the plot with plt.show()
plt.show()
```

![278](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/278.png)

> #### Scatter Plot 

```python
# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()
```

![279](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/279.png)

### Histograms
> #### Build a histogram

```python
# Build histogram with 5 bins
plt.hist(life_exp, bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins=20)

# Show and clean up again
plt.show()
plt.clf()
```

![280](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/280.png)

### Customization
> #### Additional Customizations
> > * a list col has been created for you.

```python
dict = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}
```

```python
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()
```

![281](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/281.png)

## 2. Dictionaries & Pandas
### Dictionaries, Part 1
> #### Create dictionary

```python
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo'}

# Print europe
print(europe)
```
> #### Access dictionary

```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])
```

### Dictionaries, Part 2
> #### Dictionary Manipulation (1)
> > * add

```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)
```
> #### Dictionary Manipulation (2)
> > * updata & clean

```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)
```
> #### Dictionariception
> > * Dictionaries can contain key:value pairs where the values are again dictionaries.

```python
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital':'rome', 'population':59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)
```
![282](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/282.png)

### Pandas, Part 1
> #### Dictionary to DataFrame
> > *  a list row_labels has been created
> > * setting the index attribute of cars, that you can access as cars.index

```python
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)
```
![283](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/283.png)

> #### CSV to DataFrame
> > * read_csv() call to import the CSV data didn't generate an error
> > * index_col, an argument of read_csv(), that you can use to specify which column in the CSV file should be used as a row label

```python
# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out cars
print(cars)
```
![284](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/284.png)

### Pandas, Part 2
> #### Square Brackets (1)
> > * you can index and select Pandas DataFrames in many different ways. 
> > * The simplest, but not the most powerful way, is to use square brackets.

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])
```
![285](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/285.png)

> #### Square Brackets (2)

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])
```
![286](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/286.png)

> #### loc and iloc (1)
> > * loc is label-based, which means that you have to specify rows and columns based on their row and column labels. 
> > * iloc is integer index based

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out observation for Japan
print(cars.loc[['JAP']])
print(cars.iloc[[4]])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS','EG']])
print(cars.iloc[[1,6]])
```

> #### loc and iloc (2)

```python
cars.loc['IN', 'cars_per_cap']
cars.iloc[3, 0]

cars.loc[['IN', 'RU'], 'cars_per_cap']
cars.iloc[[3, 4], 0]

cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
cars.iloc[[3, 4], [0, 1]]

cars.loc[:, 'country']
cars.iloc[:, 1]

cars.loc[:, ['country','drives_right']]
cars.iloc[:, [1, 2]]
```

## 3. Logic, Control Flow and Filtering
### Comparison Operators
> #### Equality

```python
# Comparison of booleans
print(True == False)

# Comparison of integers
print(-5 * 15 != 75)

# Comparison of strings
print("pyscript" == "PyScript")

# Compare a boolean with an integer
print(True == 1)
```

> #### Greater and less than

```python
# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print("test" <= y)

# Comparison of booleans
print(True > False)
```

> #### Compare arrays

```python
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house[ my_house >= 18]) # [ 18.  20.]

# my_house less than your_house
print(my_house[ my_house < your_house]) #  [ 20.    10.75]
```

### Boolean Operators
> #### and, or, not

```python
# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18) # False

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17) % True


# Double my_kitchen smaller than triple your_kitchen?
print(2 * my_kitchen < 3 * your_kitchen) # True
```

> #### Boolean operators with Numpy
> > * use these operators with Numpy, you will need np.logical_and(), np.logical_or() and np.logical_not()

```python
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10)) # [[False  True False  True]]

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11)) # [[False False False  True]]

```

### if, elif, else
> #### Customize further: elif

```python
# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10 :
    print("medium size, nice!")
else :
    print("pretty small.")

```

### Filtering Pandas DataFrame
> #### Driving right
> > * find all observations in cars where drives_right is True.
> > * Put the code that computes drives_right straight into the square brackets that select observations from cars
 
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr
dr = cars["drives_right"]

# Use dr to subset cars: sel
sel = cars[dr]

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)
```
> #### Cars per capita

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]


# Print medium
print(medium)
```

## 4. Loops
### while loop
> #### while loop

```python
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0 :
        offset = offset - 1
    else :
        offset = offset + 1
    print(offset)
```


### for loop
> #### Loop over a list

```python
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for area in areas :
    print(area)
```

> #### Indexes and values

```python
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate()
for x,y in enumerate(areas) :
    print("room " + str(x) + ": " + str(y))
```
> #### Loop over list of lists

```python
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch

for h in house :
    print("the " + h[0] + " is " + str(h[1]) + " sqm")
```

### Looping Data Structures, Part 1
> #### Loop over dictionary
> > * In Python 3, you need the items() method to loop over a dictionary

```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe
for key, value in europe.items() :
    print("the capital of " + key + " is " + value)
```

> #### Loop over Numpy array

```python
# Import numpy as np
import numpy as np

# For loop over np_height
for x in np_height :
    print(str(x) + " inches")

# For loop over np_baseball
for y in np.nditer(np_baseball) :
    print(str(y))
```

### Looping Data Structures, Part 2
> #### Loop over DataFrame (1)
> > * Iterating over a Pandas DataFrame is typically done with the iterrows() method

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows() :
    print(lab)
    print(row)
```

> #### Loop over DataFrame (2)
> > * you can easily select variables from the Pandas Series using square brackets

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))
```

> #### Add column (1)

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Print cars
print(cars)
```
> #### Add column (2)
> > * Compare the iterrows() version with the apply() version to get the same result in the brics

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
for lab, row in cars.iterrows() :
    cars["COUNTRY"] = cars["country"].apply(str.upper)
```

## 5. Case Study: Hacker Statistics
### Random Numbers
> #### Random float
> > * seed(): sets the random seed, so that your results are the reproducible between simulations.
> > * rand(): if you don't specify any arguments, it generates a random float between zero and one.

```python
# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())
```

> #### Roll the dice

```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1,7))

# Use randint() again
print(np.random.randint(1,7))
```

> #### Determine your next move

```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <= 5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)
```

### Random Walk
> #### The next step

```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]


    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)
```

> #### How low can you go?

```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(step - 1, 0)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

print(random_walk)
```

> #### Visualize the walk

```python
# Initialization
import numpy as np
np.random.seed(123)
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()
```
![287](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/287.png)

### Distribution
> #### Simulate multiple walks

```python
# Initialization
import numpy as np
np.random.seed(123)

# Initialize all_walks
all_walks = []


# Simulate random walk 10 times
for i in range(10) :

    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)
```

> #### Visualize all walks

```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)


# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()
```

![288](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/288.png)

> #### Plot the distribution
> > * We still have to solve the million-dollar problem: What are the odds that you'll reach 60 steps high on the Empire State Building?

```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []

# Simulate random walk 500 times
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()
```

![289](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/289.png)