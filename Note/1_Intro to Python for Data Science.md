# Intro to Python for Data Science

## 1. Python Basics
> ### Hello Python!
> #### Any comments?
> > * use the # tag

```python
# Just testing division
print(5 / 8)

# Addition works too
print(7 + 10)
``` 

> #### Python as a calculator
> > * Exponentiation: **
> > * Modulo: %

```python
# Addition and subtraction
print(5 + 5)
print(5 - 5)

# Multiplication and division
print(3 * 5)
print(10 / 2)

# Exponentiation
print(4 ** 2)

# Modulo
print(18 % 7)

# How much is your $100 worth after 7 years?
print(100 * (1.1 ** 7))
``` 

> ### Variables & Types
> #### Variable Assignment

```python
# Create a variable savings
savings = 100
# Print out savings
print(savings)
``` 

> #### Calculations with variables

```python
# Create a variable savings
savings = 100

# Create a variable factor
factor = 1.10

# Calculate result
result = savings * factor ** 7

# Print out result
print(result)
``` 

> #### Other variable types
> > * int, or integer: a number without a fractional part
> > * float, or floating point: a number that has both an integer and fractional part
> > * str, or string: a type to represent text. You can use single or double quotes to build a string.
> > * bool, or boolean: a type to represent logical values. Can only be True or False (the capitalization is important!).

> #### Guess the type
> > * use the type() function.

> #### Operations with other types

```python
# Several variables to experiment with
savings = 100
factor = 1.1
desc = "compound interest"

# Assign product of factor and savings to year1
year1 = savings * factor

# Print the type of year1
print(type(year1))
# <class 'float'>

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(doubledesc)
# compound interestcompound interest
``` 

> #### Type conversion
> > * Using the + operator to paste together two strings can be very useful in building custom messages
> > * you'll need str(), to convert a value into a string. str(savings)
> > * Similar functions such as int(), float() and bool() will help you convert Python values into any type.

```python
# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# I started with $100 and now have $194.87171000000012. Awesome!

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)
``` 

## 2. Python Lists
> ### Lists, what are they?
> #### Create a list
> > * list is a compound data type

```python
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]

# Print areas
print(areas)
``` 

> #### Create list with different types

```python
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom",  bed, "bathroom", bath]

# Print areas
print(areas)
``` 
> #### List of lists

```python
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))
``` 

> ### Subsetting lists
> #### Subset and conquer

```python
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[5])
``` 

> #### Subset and calculate

```python
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[7]

# Print the variable eat_sleep_area
print(eat_sleep_area)
``` 

> #### Slicing and dicing
> > * my_list[start:end]
> > * start index will be included, while the end index is not.

```python
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs = areas[0:6]

# Use slicing to create upstairs
upstairs = areas[6:]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)
``` 
![275](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/275.png)

> #### Slicing and dicing (2)

```python
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[6:]
``` 

> #### Subsetting lists of lists

```python
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]

x[2][0] # 'g'
x[2][:2] # '['g', 'h']'
``` 

> ### List Manipulation
> #### Replace list elements

```python
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"
``` 

> #### Extend a list

```python
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas +  ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]
```
> #### Delete list elements
> > * the del statement

```python
x = ["a", "b", "c", "d"]
del(x[1])
```

> #### Inner workings of lists

```python
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)
```

## 3. Functions and Packages
> ### Functions
> #### Help!
> > *  ask for information about a function with another function: help()

```python
help(max)
?max
```

> #### Multiple arguments
> > * see that sorted() takes three arguments: iterable, key and reverse

```python
# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, key=None, reverse= True)

# Print out full_sorted
print(full_sorted)
```

> ### Methods
> #### List Methods
> > * index(), to get the index of the first element of a list that matches its input and
> > * count(), to get the number of times an element appears in a list.
> > * append(), that adds an element to the list it is called on,
> > * remove(), that removes the first element of a list that matches the input, and
> > * reverse(), that reverses the order of the elements in the list it is called on.

> ### Packages
> #### Import package

```python
# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2*math.pi*r

# Calculate A
A = math.pi*r**2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))
```

> #### Selective import
> > * from math import pi 

## 4. NumPy
> ### NumPy
> #### Your First NumPy Array

```python
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))
```

> #### Baseball players' height

```python
# height is available as a regular list

# Import numpy
import numpy as np

# Create a numpy array from height: np_height
np_height = np.array(height)

# Print out np_height
print(np_height)

# Convert np_height to m: np_height_m
np_height_m = np_height * 0.0254

# Print np_height_m
print(np_height_m)
```

> #### Baseball player's BMI

```python
# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254

# Create array from weight with correct units: np_weight_kg
np_weight_kg = np.array(weight) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m**2

# Print out bmi
print(bmi)
```

> #### Lightweight baseball players

```python
# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])
```

> #### NumPy Side Effects
> > * First of all, numpy arrays cannot contain elements with different types. 
> > * Second, the typical arithmetic operators, such as +, -, * and / have a different meaning for regular Python lists

```python
np.array([True, 1, 2]) + np.array([3, 4, False]) # array([4, 5, 2])

```

> #### Subsetting NumPy Arrays

```python
# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Store weight and height lists as numpy arrays
np_weight = np.array(weight)
np_height = np.array(height)

# Print out the weight at index 50
print(np_weight[50])

# Print out sub-array of np_height: index 100 up to and including index 110
print(np_height[100:111])
```

> ### 2D NumPy Arrays
> #### Your First 2D NumPy Array

```python
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)
```

> #### Baseball data in 2D form

```python
# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)  # (1015, 2)
```
> #### Subsetting 2D NumPy Arrays

```python
# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123][0])
```

> #### 2D Arithmetic

```python
# baseball is available as a regular list of lists
# updated is available as 2D numpy array

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)
```

> ### NumPy: Basic Statistics
> #### Average versus median

```python
# np_baseball is available

# Import numpy
import numpy as np

# Create np_height from np_baseball
np_height = np.array(np_baseball[:,0])

# Print out the mean of np_height
print(np.mean(np_height))

# Print out the median of np_height
print(np.median(np_height))

```

> #### Explore the baseball data

```python
# np_baseball is available

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))
```

![276](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/276.png)

> #### Blend it all together

```python
# heights and positions are available as lists

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions = np.array(positions)
np_heights = np.array(heights)


# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK']

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))
```

![277](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/277.png)