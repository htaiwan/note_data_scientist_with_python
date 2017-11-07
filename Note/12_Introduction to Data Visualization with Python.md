# Introduction to Data Visualization with Python

## 1. Customizing plots
> * a review of basic plotting with Matplotlib
> * customizing plots using Matplotlib. This includes overlaying plots, making subplots, controlling axes, adding legends and annotations, and using different plot styles

### Plotting multiple graphs
> #### Multiple plots on single axis
> > * issue two plt.plot() commands to draw line plots of different colors on the same set of axes 

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color= 'red')

# Display the plot
plt.show()
```

![145](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/145.png)

> #### Using axes()
> > * plt.axes([xlo, ylo, width, height]), a set of axes is created and made active with lower corner at coordinates (xlo, ylo) of the specified width and height

```python
# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, 0.05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()
```
![146](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/146.png)

> #### Using subplot() (1)
> > * The command plt.axes() requires a lot of effort to use well because the coordinates of the axes need to be set manually.
> > * A better alternative is to use plt.subplot() to determine the layout automatically.
> > * plt.subplot(m, n, k) to make the subplot grid of dimensions m by n and to make the kth subplot active (subplots are numbered starting from 1 row-wise from the top left corner of the subplot grid)

```python
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()
```
![147](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/147.png)

> #### Using subplot() (2)

```python
# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()
```

![148](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/148.png)


### Customizing axes
> #### Using xlim(), ylim()
> > * These commands allow you to either zoom or expand the plot or to set the axis ranges to include important values (such as the origin).

```python
# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim(1990, 2010)

# Set the y-axis range
plt.ylim(0, 50)

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')
```
![149](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/149.png)

> #### Using axis()
> > * how you can pass a 4-tuple to plt.axis() to set limits for both axes at once. 
> > * For example, plt.axis((1980,1990,0,75)) would set the extent of the x-axis to the period between 1980 and 1990, and would set the y-axis extent from 0 to 75% degrees award.

```python
# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990, 2010, 0, 50))

# Show the figure
plt.show()

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')
```

![150](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/150.png)

### Legends, annotations, and styles
> #### Using legend()
> > * Legends are useful for distinguishing between multiple datasets displayed on common axes.
> > * Using the keyword argument label in the plotting function associates a string to use in a legend.
> > * request a legend using plt.legend(). Specifying the keyword argument loc determines where the legend will be placed

```python
# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')

# Add a legend at the lower center
plt.legend(loc='lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()
```
![151](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/151.png)

> #### Using annotate()
> > * It is often useful to annotate a simple plot to provide context. 
> > * This makes the plot more readable and can highlight specific aspects of the data. Annotations like text and arrows can be used to emphasize specific observations.

```python
# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Add a black arrow annotation
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()
```

![152](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/152.png)

> #### Modifying styles
> > * Matplotlib comes with a number of different stylesheets to customize the overall look of different plots. 
> > * To activate a particular stylesheet you can simply call plt.style.use()

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()
```
![153](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/153.png)

## 2. Plotting 2D arrays
> * visualizing two-dimensional arrays
> * presentation, and orientation of grids for representing two-variable functions followed by discussions of pseudocolor plots, contour plots, color maps, two-dimensional histograms, and images.

### Working with 2D arrays
> #### Generating meshes
> > *  use the meshgrid function in NumPy to generate 2-D arrays which you will then visualize using plt.imshow()
> > * Generate two one-dimensional arrays u and v using np.linspace()
> > * using np.meshgrid(). The resulting arrays should have shape (41,21)

```python
# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u, v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()

# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')
```
![154](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/154.png)

> #### Array orientation

![155](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/155.png)

![156](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/156.png)

> > * A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])

### Visualizing bivariate functions
> #### Contour & filled contour plots
> > * Although plt.imshow() or plt.pcolor() are often used to visualize a 2-D array in entirety,
> > * there are other ways of visualizing such data without displaying all the available sample values. One option is to use the array to compute contours that are visualized instead.


```python
# Generate a default contour map of the array Z
plt.subplot(2,2,1)
plt.contour(X, Y, Z)

# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)

# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)

# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)

# Improve the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
```

![157](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/157.png)

> #### Modifying colormaps
> > * When displaying a 2-D array with plt.imshow() or plt.pcolor(), the values of the array are mapped to a corresponding color. The set of colors used is determined by a colormap which smoothly maps values to colors
> > * the colormap from the default 'jet' colormap used by matplotlib

```python
# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()
```

![158](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/158.png)

### Visualizing bivariate distributions
> #### Using hist2d()
> > * plt.hist2d(). You specify the coordinates of the points using plt.hist2d(x,y) assuming x and y are two vectors of the same length.
> > 
> > > * You can specify the number of bins with the argument bins=(nx, ny) where nx is the number of bins to use in the horizontal direction and ny is the number of bins to use in the vertical direction.
> > 
> > > * You can specify the rectangular region in which the samples are counted in constructing the 2D histogram. The optional parameter required is range=((xmin, xmax), (ymin, ymax)) where
> > 
> > > * xmin and xmax are the respective lower and upper limits for the variables on the x-axis and
> > 
=
> > > * ymin and ymax are the respective lower and upper limits for the variables on the y-axis. Notice that the optional range argument can use nested tuples or lists.

```python
# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins=(20, 20), range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()
```
![159](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/159.png)

> #### Using hexbin()
> > *  the function plt.hexbin() uses hexagonal bins
> > 
> > > * The optional gridsize argument (default 100) gives the number of hexagons across the x-direction used in the hexagonal tiling. If specified as a list or a tuple of length two, gridsize fixes the number of hexagon in the x- and y-directions respectively in the tiling.
> > 
> > > * The optional parameter extent=(xmin, xmax, ymin, ymax) specifies rectangular region covered by the hexagonal tiling. In that case, xmin and xmax are the respective lower and upper limits for the variables on the x-axis and ymin and ymax are the respective lower and upper limits for the variables on the y-axis

```python
# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize=(15, 12), extent=(40, 235, 8, 48))

           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()
```

![160](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/160.png)

### Working with images
> #### Loading, examining images

```python
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape) #  (480, 480, 3)

# Display the image
plt.imshow(img)

# Hide the axes
plt.axis('off')
plt.show()
```
> #### Pseudocolor plot from image data
> > *  In many situations, an image may be processed and analysed in some way before it is visualized in pseudocolor, also known as 'false' color.
> > * perform a simple analysis using the image showing an astronaut as viewed from space

```python
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis=2)

# Print the shape of the intensity
print(intensity.shape)

# Display the intensity with a colormap of 'gray'
plt.imshow(intensity, cmap='gray')

# Add a colorbar
plt.colorbar()

# Hide the axes and show the figure
plt.axis('off')
plt.show()
```
![161](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/161.png)

> #### Extent and aspect
> > * The ratio of the displayed width to height is known as the image aspect and the range used to label the x- and y-axes is known as the image extent. The default aspect value of 'auto' keeps the pixels square and the extents are automatically computed from the shape of the array if not specified otherwise

```python
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)


# Improve spacing and display the figure
plt.tight_layout()
plt.show()
```
![162](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/162.png)

> #### Rescaling pixel intensities
> > * Sometimes, low contrast images can be improved by rescaling their intensities
> > * you will do a simple rescaling (remember, an image is NumPy array) to translate and stretch the pixel intensities so that the intensities of the new image fill the range from 0 to 255.

```python
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))

# Rescale the pixels: rescaled_image
rescaled_image = 256*(image-pmin) / (pmax-pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." % 
      (rescaled_image.min(), rescaled_image.max()))

# Display the original image in the top subplot
plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)

# Display the rescaled image in the bottom subplot
plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image)

plt.show()

#   The smallest & largest pixel intensities are 104 & 230.
#   The rescaled smallest & largest pixel intensities are 0.0 & 256.0.
```
![163](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/163.png)

## 3. Statistical plots with Seaborn
> * a high-level tour of the Seaborn plotting library for producing statistical graphics in Python.
> * for computing and visualizing linear regressions as well as tools for visualizing univariate distributions
> * Two types of contour plot supported by Matplotlib are plt.contour() and plt.contourf() where the former displays the contours as lines and the latter displayed filled areas between contours

### Visualizing regressions
> #### Simple linear regressions
> > * seaborn provides a convenient interface to generate complex and great-looking statistical plots
> > * One of the simplest things you can do using seaborn is to fit and visualize a simple linear regression between two variables using sns.lmplot().

```python
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)

# Display the plot
plt.show()
```

![164](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/164.png)

> #### Plotting residuals of a regression
> > * Seaborn provides sns.residplot() for that purpose, visualizing how far datapoints diverge from the regression line.

```python
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')

# Display the plot
plt.show()
```

![165](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/165.png)

> #### Higher-order regressions
> > * using sns.regplot() (the function sns.lmplot() is a higher-level interface to sns.regplot()). 
> > * To force a higher order regression, you need to specify the order parameter. Here, it should be 2.

```python
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='order 1')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, order=2, color='green', scatter=None, label='order 2')

# Add a legend and display the plot
plt.legend(loc='upper right')
plt.show()
```
![166](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/166.png)

> #### Grouping linear regressions by hue
> > * Using the hue argument, you can specify a categorical variable by which to group data observations.

```python
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='hp', data=auto, hue='origin', palette='Set1')

# Display the plot
plt.show()
```
![167](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/167.png)

> #### Grouping linear regressions by row or column
> > * The sns.lmplot() accepts the arguments row and/or col to arrangements of subplots for regressions.

```python
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x='weight', y='hp', data=auto, row='origin')

# Display the plot
plt.show()
```
![168](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/168.png)

### Visualizing univariate distributions
> #### Constructing strip plots
> > * Regressions are useful to understand relationships between two continuous variables. 
> > * Often we want to explore how the distribution of a single continuous variable is affected by a second categorical variable.
> > * The strip plot is one way of visualizing this kind of data. It plots the distribution of variables for each category as individual datapoints. 

```python
# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, jitter=True, size=3)

# Display the plot
plt.show()
```
![169](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/169.png)

> #### Constructing swarm plots
> > * An alternative is provided by the swarm plot (sns.swarmplot()), which is very similar but spreads out the points to avoid overlap and provides a better visual overview of the data.

```python
# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x='cyl', y='hp', data=auto) 

# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x='hp', y='cyl', data=auto, hue='origin', orient='h') 

# Display the plot
plt.show()
```
![170](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/170.png)

> #### Constructing violin plots
> > * Both strip and swarm plots visualize all the datapoints. For large datasets, this can result in significant overplotting
> > * Therefore, it is often useful to use plot types which reduce a dataset to more descriptive statistics and provide a good summary of the data. Box and whisker plots are a classic way of summarizing univariate distributions but seaborn provides a more sophisticated extension of the standard box plot, called a violin plot.

```python
# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)

# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, inner=None, color='lightgray')

# Overlay a strip plot on the violin plot
sns.stripplot(x='cyl', y='hp', data=auto, jitter=True, size=1.5)

# Display the plot
plt.show()
```
![171](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/171.png)

### Visualizing multivariate distributions
> #### Plotting joint distributions (1)
> > * Seaborn's sns.jointplot() provides means of visualizing bivariate distributions

```python
# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x='hp', y='mpg', data=auto)

# Display the plot
plt.show()
```
![172](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/172.png)

> #### Plotting joint distributions (2)
> > * kind='scatter' uses a scatter plot of the data points
> > * kind='reg' uses a regression plot (default order 1)
> > *  kind='resid' uses a residual plot
> > *  kind='kde' uses a kernel density estimate of the joint distribution
> > *  kind='hex' uses a hexbin plot of the joint distribution

```python
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')

# Display the plot
plt.show()
```
![173](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/173.png)

> #### Plotting distributions pairwise (1)
> > * The function sns.pairplot() constructs a grid of all joint plots pairwise from all pairs of (non-categorical) columns in a DataFrame. 

```python
# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions from the DataFrame 
sns.pairplot(auto)

# Display the plot
plt.show()
```
![174](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/174.png)

![175](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/175.png)

> #### Plotting distributions pairwise (2)
> > * in the off-diagonal subplots. You will do this with the argument kind='reg' (where 'reg' means 'regression'). Another option for kind is 'scatter' (the default) 
> > * this with the keyword argument hue specifying the 'origin'

```python
# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(hue='origin', kind='reg', data=auto)

# Display the plot
plt.show()
```
![176](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/176.png)

![177](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/177.png)

## 4. Analyzing time series and images
> *  customizing plots of stock data, generating histograms of image pixel intensities, and enhancing image contrast through histogram equalization.

### Visualizing time series
> #### Multiple time series on common axes

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color='magenta', label='MSFT')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

# Display the plot
plt.show()
```
![178](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/178.png)

> #### Multiple time series slices (1)

```python
# Import matplotlib.pyplot
# Plot the series in the top subplot in blue
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')

# Slice aapl from '2007' to '2008' inclusive: view
view = aapl['2007':'2008']

# Plot the sliced data in the bottom subplot in black
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()
plt.show()
```

![179](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/179.png)

> #### Multiple time series slices (2)

```python
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the sliced series in the top subplot in red
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.plot(view, color='red')

# Reassign the series by slicing the month January 2008
view = aapl['2008-01']

# Plot the sliced series in the bottom subplot in green
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL: Jan. 2008')
plt.plot(view, color='green')

# Improve spacing and display the plot
plt.tight_layout()
plt.show()
```

![180](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/180.png)

> #### Plotting an inset view

```python
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the entire series 
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

# Specify the axes
plt.axes([0.25, 0.5, 0.35, 0.35])

# Plot the sliced series in red using the current axes
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()
```

![181](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/181.png)

### Time series with moving windows
> #### Plotting moving averages
> > * The time series mean_30, mean_75, mean_125, and mean_250 have been computed

```python
# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2, 2, 1)
plt.plot(mean_30, 'green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 75-day moving average in the top right subplot in red
plt.subplot(2, 2, 2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

# Plot the 125-day moving average in the bottom left subplot in magenta
plt.subplot(2, 2, 3)
plt.plot(mean_125, 'magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2, 2, 4)
plt.plot(mean_250, 'cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')

# Display the plot
plt.show()

```

![182](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/182.png)

### Histogram equalization in images
> #### Extracting a histogram from a grayscale image

```python
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,255))
plt.title('Normalized histogram')
plt.hist(pixels, bins=64, range=(0,256), normed=True, color='red', alpha=0.4)

# Display the plot
plt.show()
```
![183](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/183.png)

> #### Cumulative Distribution Function from an image histogram

```python
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
pdf = plt.hist(pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0,256),
               normed=True, cumulative=True,
               color='blue', alpha=0.4)
               
# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0,256))
plt.grid('off')
plt.title('PDF & CDF (original image)')
plt.show()
```
![184](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/184.png)

> #### Equalizing an image histogram

```python
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

# Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)

# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image, cmap='gray')

# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid('off')

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
plt.show()
```
![185](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/185.png)

> #### Extracting histograms from a color image

```python
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Display image in top subplot
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image)

# Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2,1,2)
plt.title('Histograms from color image')
plt.xlim((0,256))
plt.hist(red_pixels, bins=64, normed=True, color='red', alpha=0.2)
plt.hist(blue_pixels, bins=64, normed=True, color='blue', alpha=0.2)
plt.hist(green_pixels, bins=64, normed=True, color='green', alpha=0.2)

# Display the plot
plt.show()
```
![186](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/186.png)

> #### Extracting bivariate histograms from a color image

```python
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Extract RGB channels and flatten into 1-D array
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid('off') 
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(red_pixels, green_pixels, bins=(32,32))

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(green_pixels, blue_pixels, bins=(32,32))

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(blue_pixels, red_pixels, bins=(32,32))

# Display the plot
plt.show()
```
![187](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/187.png)
