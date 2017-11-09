# Interactive Data Visualization with Bokeh

## 1. Basic plotting with Bokeh
> * An introduction to basic plotting with Bokeh

### Plotting with glyphs
> #### What are glyphs?
> > * In Bokeh, visual properties of shapes are called glyphs. The visual properties of these glyphs such as position or color can be assigned single values, for example x=10 or fill_color='red'

> #### A simple scatter plot
> > * The x-axis data has been loaded for you as fertility and the y-axis data has been loaded as female_literacy
> > * create a figure, assign x-axis and y-axis labels, and plot female_literacy vs fertility using the circle glyph.

```python
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)
```
![188](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/188.png)

> #### A scatter plot with different shapes
> > * plot the Latin America data with the circle() glyph, and the Africa data with the x() glyph.

```python
# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)
```
![189](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/189.png)

> #### Customizing your scatter plots
> > * he three most important arguments to customize scatter glyphs are color, size, and alpha

```python
# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)
```
![190](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/190.png)


### Additional glyphs
> #### Lines
> > * draw lines on Bokeh plots with the line() glyph function.
> > * date is a list of datetime objects to plot on the x-axis and price is a list of prices to plot on the y-axis.

```python
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date, price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)
```

![191](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/191.png)

> #### Lines and markers
> > * Lines and markers can be combined by plotting them separately using the same data points.

```python
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)
```

![192](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/192.png)

> #### Patches
> > * extended geometrical shapes can be plotted by using the patches() glyph function. 
> > * The patches glyph takes as input a list-of-lists collection of numeric values specifying the vertices in x and y directions of each distinct patch to plot

```python
# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color='white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)
```

![193](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/193.png)

### Data formats
> #### Plotting data from NumPy arrays
> > * generate NumPy arrays using np.linspace() and np.cos() and plot them using the circle glyph.
> > * np.linspace() is a function that returns an array of evenly spaced numbers over a specified interval. 
> > * For example, np.linspace(0, 10, 5) returns an array of 5 evenly spaced samples calculated over the interval [0, 10]

```python
# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0, 5, 100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x, y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)
```

![194](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/194.png)

> #### Plotting data from Pandas DataFrames
> > * create Bokeh plots from Pandas DataFrames by passing column selections to the glyph functions.

```python
# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)
```

![195](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/195.png)

> #### The Bokeh ColumnDataSource
> > * The ColumnDataSource is a table-like data object that maps string column names to sequences (columns) of data. It is the central and most common data structure in Bokeh.
> > * All columns in a ColumnDataSource must have the same length.
> > * can create a ColumnDataSource object directly from a Pandas DataFrame by passing the DataFrame to the class initializer.

```python
# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time',  source=source, color='color', size=8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)
```
![196](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/196.png)

### Customizing glyphs
> #### Selection and non-selection glyphs
> > * add the box_select tool to a figure and change the selected and non-selected circle glyph properties so that selected glyphs are red and non-selected glyphs are transparent blue.
> > * use the ColumnDataSource object of the Olympic Sprint dataset you made in the last exercise. It is provided to you with the name source.

```python
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source=source, selection_color='red', nonselection_fill_color='blue', nonselection_alpha=0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)
```
![197](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/197.png)

> #### Hover glyphs
> > * add a circle glyph that will appear red when the mouse is hovered near the data points. You will also add a customized hover tool object to the plot.

```python
# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)
```
![198](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/198.png)

> #### Colormapping
> > * using the CategoricalColorMapper to color each glyph by a categorical property.

```python
#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)
```
![199](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/199.png)

## 2. Layouts, Interactions, and Annotations
> * Learn how to combine mutiple Bokeh plots into different kinds of layouts on a page

### Introduction to layouts
> #### Creating rows of plots
> > * Layouts are collections of Bokeh figure objects.
> > * By using the row() method, you'll create a single layout of the two figures.

```python
# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)

# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)
```
![200](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/200.png)

> #### Creating columns of plots
> > * use the column() function to create a single column layout of the two plots 

```python
# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)


# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)
```
> #### Nesting rows and columns of plots
> > * use the column() and row() functions to make a two-row layout where the first row will have only 
> > * using the sizing_mode argument, you can scale the widths to fill the whole figure.

```python
# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)
```
![201](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/201.png)

### Advanced layouts
> #### Creating gridded layouts
> > * Regular grids of Bokeh plots can be generated with gridplot

```python
# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)
```
![202](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/202.png)

> #### Starting tabbed layouts
> > * Tabbed layouts can be created in Pandas by placing plots or layouts in Panels.
> > * using Tabs() and assign the tabs keyword argument to your list of Panels

```python
# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

```
![203](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/203.png)

### Linking plots together
> #### Linked axes
> > * Linking axes between plots is achieved by sharing range objects.

```python
# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)

```
![204](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/204.png)

> #### Linked brushing
> > * By sharing the same ColumnDataSource object between multiple plots, 
> > * selection tools like BoxSelect and LassoSelect will highlight points in both plots that share a row in the ColumnDataSource
> > Use your mouse to drag a box or lasso around points in one figure, and notice how points in the other figure that share a row in the ColumnDataSource also get highlighted.

```python
# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)

# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)
```
![205](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/205.png)

### Annotations and guides
> #### How to create legends
> > * Legends can be added to any glyph by using the legend keyword argument.

```python
# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)
```
![206](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/206.png)

> #### Hover tooltips for exposing details
> > * hover tools, certain pre-defined fields such as mouse position or glyph index can be accessed
> > * Working with the HoverTool is easy for data stored in a ColumnDataSource.
> > * This is done by assigning the tooltips keyword argument to a list-of-tuples specifying the label and the column of values from the ColumnDataSource using the @ operator

```python
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips=[('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)

```
![207](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/207.png)

## 3. Building interactive apps with Bokeh
> * Bokeh server applications let you connect all of the powerful Python libraries for analytics and data science, such as NumPy and Pandas

### Introducing the Bokeh Server
> #### Understanding Bokeh apps
> > * The main purpose of the Bokeh server is to synchronize python objects with web applications in a browser, so that rich, interactive data applications can be connected to powerful PyData libraries such as NumPy, SciPy, Pandas, and scikit-learn

> #### Using the current document
> > * begins with importing the curdoc, or "current document", function from bokeh.io
> > * running a Bokeh app using the bokeh serve command line tool

```python
# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line([1,2,3,4,5], [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)
```
![208](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/208.png)

> #### Add a single slider
> > *  create a single slider, use it to create a widgetbox layout, and then add this layout to the current document.

```python
# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)
```
![209](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/209.png)

> #### Multiple sliders in one document

```python
# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)

# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)
```

### Connecting sliders to plots
> #### How to combine Bokeh models into layouts
> > * making a Bokeh application that has a simple slider and plot, that also updates the plot based on the slider

```python
# Create ColumnDataSource: source
source = ColumnDataSource(data={'x':x, 'y':y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)
```
![210](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/210.png)

> #### Learn about widget callbacks
> > * how to use widget callbacks to update the state of a Bokeh application, and in turn, the data that is presented to the user.
> > * use the slider's on_change() function

```python
# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)
```

### Updating plots from dropdowns
> #### Updating data sources from dropdown callbacks
> > * learn to update the plot's data using a drop down menu instead of a slider

```python
# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)
```

> #### Synchronize two dropdowns
> > * using a dropdown callback to update another dropdown's options

```python
# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A' 
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)
```

### Buttons
> #### Button widgets
> > * to create a button and use its on_click() method to update a plot.

```python
# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x': x, 'y': y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)
```
> #### Button styles
> > * using CheckboxGroup, RadioGroup, and Toggle to add multiple Button widgets with different styles.

```python
# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(label='Toggle button', button_type='success')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))
```
![211](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/211.png)

## 4. Putting It All Together! A Case Study
> * build a more sophisticated Bokeh data exploration application from the ground up, based on the famous Gapminder data set.

### Time to put it all together!
> #### Some exploratory plots of the data
> > * by making a simple plot of Life Expectancy vs Fertility for the year 1970.
> > * prepare a ColumnDataSource object with the fertility, life and Country columns, where you only select the rows with the index value 1970.

```python
# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)
```
![212](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/212.png)

### Starting the app
> #### Beginning with just a plot
> > * make the ColumnDataSource object, prepare the plot, and add circles for Life expectancy vs Fertility. 
> > * also set x and y ranges for the axes.

```python
# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
    'pop'     : (data.loc[1970].population / 20000000) + 2,
    'region'  : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'
```
![213](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/213.png)

> #### Enhancing the plot with some shading
> > * enhance it by coloring each circle glyph by continent.
> > * prepare a ColorMapper, and add it to the circle glyph.

```python
# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'
```

![214](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/214.png)

> #### Adding a slider to vary the year
> > * create an update_plot() function and associate it with a slider to select values between 1970 and 2010.

```python
# Import the necessary modules
from bokeh.layouts import widgetbox, row
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # set the `yr` name to `slider.value` and `source.data = new_data`
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data  = new_data


# Make a slider object: slider
slider = Slider(title='Year', start=1970, end=2010, step=1, value=1970)

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)
```

> #### Customizing based on user input
> > * specifying placeholders with the % keyword. For example, if you have a string company = 'DataCamp', you can use print('%s' % company) to print DataCamp

```python
# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(title='Year', start=1970, end=2010, step=1, value=1970)

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)
```

### Adding more interactivity to the app
> #### Adding a hover tool
> > * adding a hover tool to drill down into data column values and display more detailed information about each scatter point.

```python
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country','@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)
```
![215](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/215.png)

> #### Adding dropdowns to the app
> > * add dropdowns for interactively selecting different data features.

```python
# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)
```
![216](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/216.png)