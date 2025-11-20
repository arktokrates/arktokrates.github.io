---
author: "Bernhard Fuchs"
layout: default
permalink: /KI/Data-Science/
last_modified_at: 2025-11-20
---


# Essentials for Data Science


Some notes on essential data structures and functions of Python for Data Science.



## Contents

1. [General Python](#general-python)

	1. [Important data structures](#important-data-structures)

	1. [Lists](#lists)

	1. [Sets](#sets)

	1. [Tuples](#tuples)

	1. [Dictionaries](#dictionaries)

1. [Strings](#strings)

1. [Reading and writing CSV files](#reading-writing-csv-files)	

1. [Important packages for data science](#important-packages-data-science)

    1. [NumPy](#numpy)

	    1. [Arithmetic operations with arrays](#arithmetic-operations-with-arrays)

	    1. [Conditional logic in arrays](#conditional-logic-in-arrays)

	    1. [Common mathematical and statistical functions](#common-functions)

	    1. [Accessing and slicing arrays](#accessing-slicing-arrays)

	    1. [Reading and writing files in NumPy](#reading-writing-numpy)

	    1. [Simulations with Numpy](#simulations-numpy)


    1. [Pandas](#pandas)

		1. [Pandas Series](#pandas-series)

		1. [Pandas Dataframe](#pandas-sataframe)

		1. [Pandas Panel](#pandas-panel)

		1. [Common functions in Pandas](#common-functions-in-pandas)

		1. [Statistical functions in Pandas](#statistical-functions-in-pandas)

		1. [Date and Timedelta](#date-and-timedelta)

		1. [IO tools](#io-tools)

		1. [Categorical data](#categorical-data)

		1. [Working with text data (string data)](#working-with-text-data)

		1. [Iteration](#iteration)

		1. [Sorting in Pandas](#sorting-in-pandas)

		1. [Plotting with Pandas](#plotting-with-pandas)

		1. [Data Analysis](#data-analysis)

		1. [Types of data](#types-of-data)

		1. [Working with Data](#working-with-data)

		1. [Importing and exporting data in Python](#importing-and-exporting-data-in-Python)

		1. [Regular expressions in Python](#regular-expressions-in-python)

		1. [Accessing databases in Python](#accessing-databases-in-python)

	1. [Data Wrangling](#data-wrangling)

		1. [Pandorable Code (Idiomatic Pandas Code)](#pandorable-code)

		1. [Loading, Indexing, and Reindexing](#loading-indexing-and-reindexing)

		1. [Merging dataframes](#merging-dataframes)

		1. [Memory optimization in Python](#memory-optimization-in-python)

		1. [Data pre-processing](#data-pre-processing)

		1. [Describing data](#describing-data)

		1. [Data binning: formatting and normalization](#data-binning-formatting-and-normalization)

	1. [Data Visualization](#cata-visualization)

		1. [Principles of information visualization](#principles-of-information-visualization)

		1. [Data Visualization library Mathplotlib](#data-visualization-library-mathplotlib)

		1. [Data Visualization library Seaborn](#data-visualization-library-seaborn)

		1. [Data Visualization library Plotly](#data-visualization-library-plotly)

		1. [Data Visualization library Bokeh](#data-visualization-library-bokeh)

    1. [SciPy](#scipy)

    1. [Statsmodels](#statsmodels)







## General Python {#general-python}


### Important data structures {#important-data-structures}


#### Lists {#lists}

A set is an ordered, indexed, and changeable collection with possible duplicates in square brackets.

	list = ['text', 100, 9.8, 12+2j]
	Alternatively with using the list constructor: list = list('text', 100, 9.8, 12+2j)

* in square brackets: [ ]
* different types of data values
* mutable, less performative (than tuple)


##### Access to list members

	list[2]   // 100
	list[1:3]   // range (slicing): 100, 9.8
	list[-2: ]   // range open to the right, negative index (last - 2): 9.8, 12+2j


##### List operations

Multiplication:

	list * 2   // ['text', 100, 9.8, 12+2j, 'text', 100, 9.8, 12+2j]

Concatenation:

	list2 = [3, 5]
	list + list2   // ['text', 100, 9.8, 12+2j, 3, 5]


##### List functions

	len(list)   // 8

	list.pop()   // 12+2j (last element)
	list.pop(2)   // 9.8 (element at specific index)

	list.append('new text')   // ['text', 100, 'new text']
	list.append([2, 'Now'])   // ['text', 100, 'new text', [2, 'Now']]

	list.extend([32, 'Then'])   // ['text', 100, 'new text', [2, 'Now'], 32, 'Then']


	list.count('text')  // 1 

	list.insert(1, 'Blog')   // ['text', 'Blog', 100, 'new text', [2, 'Now'], 32, 'Then']

	list3 = ['First', 'Middle', 'Last']
	list3.reverse()   // ['Last', 'Middle', 'First']

	list3.sort()   // ['First', 'Last', 'Middle']
	list3.sort(reverse=True)   // ['Middle', 'Last', 'First']
	list3.sort(key = mySortFunction)

	list(set(list).intersection(list2))
	list(set(list1) & set(list2))

	list4 = [i ** 2 for i in range(5)]   // [0, 1, 4, 9, 16, 25] => list comprehension


###### Zipping and unzipping

	places  = ["Berne", "Paris", "Rome"]
	temperatures = [20, 23, 30]

	mappedValues = zip(places, temperatures)
	list(mappedValues)   // [('Berne', 20), ('Paris', 23), ('Rome', 30)]

	places, temperatures = zip(*mappedValues)   // Unzipping to separate lists




#### Sets {#sets}

A set is an unordered and unindexed collection without any duplicate value in curly brackets. It can hold different types of data values and is immutable (items are unchangeable, but you can add new items or remove items). A set is more performative than list.


	myset = {1, 3, 5, 7}


	len(mySet)   // 4

	fruitSet = set(("apple", "banana", "cherry"))   // set() constructor




#### Tuples {#tuples}

A tuple is an ordered, indexed, and unchangeable collection in parentheses with possible duplicates (more performative than lists).

	myTuple = (1, 3, 3, 5)

	len(myTuple)   // 4

	t = ('apple',)   // Define a tuple of length 1 (comma)




#### Dictionaries {#dictionaries}

A dictionary is an ordered and changeable collection without duplicates (of hash-table type, consisting of key-value pairs), in curly braces: {}.


	myDictionary = {i : i ** 2 for i in range(5)}   // {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25} => list comprehension

	myDictionary.keys()   // dict_keys([0, 1, 2, 3, 4, 5])
	myDictionary.values()   // dict_values([0, 1, 4, 9, 16, 25])
	myDictionary.items()   // dict_items([(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)])   // list of tuples

	myDictionary[6] = 36   // assignment

	dict2 = myDictionary.copy()   // returns a shallow copy (only copying the address of the object, not its content)

	import copy
	dict3 = copy.deepcopy(myDictionary)   // returns a deep copy (copy the contents, not just the address; a fully independent clone of the original object and all of its children)

	myDictionary.get(3)   //  9

	7 in myDictionary   // False

	dictNew = {7: 49, 8: 64}
	myDictionary.update(dictNew)   // {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

	myDictionary.pop(8)   // 64   => {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49}
	myDictionary.pop(8, 'null')   // null (default value)

	myDictionary.clear()   // removes all elements from dictionary




### Strings {#strings}

	s = 'Hello'
	s = "Hello"

	s = '''This is
	a multiline string'''

	s = 'He said, \'Hi\'.'    // or: "He said, 'Hi'.""

Access to string values via index:
```Python
fragment = s[1]   // 'h'
s[-1]   // 'g'
```



#### String functions

	len(s)   // 25

Outputting:
```Python
for i in range(len(s)):
    print(s[i])
```

	'is' in s   // True
	'IS' not in s   // True

Slicing:
```s[0:4]```   // 'This'

Reverting a string:
```Python
s = "Hello World"
s[-1::-1]   // 'dlroW olleH'
```


	s.upper()   // 'HELLO WORLD'
	s.lower()   // 'hello world'
	s.split(' ')   // ['Hello', 'World']
	s.replace('d', 't')   // 'Hello Worlt'

	Concatenation:
	s + ", now!"   // 'Hello Worlt, now!'

	Formatting
	print('{} Welcome to Switzerland'.format(', Tester!'))   // 'Welcome to Switzerland, Tester!'





### Reading and writing of CSV files {#reading-writing-csv-files}

```Python
import csv

csvFile = open('CSVFile.csv')
csvReader = csv.reader(csvFile)

for line in csvReader:
	print(line)
```

#### More useful reading example:
```Python
with open('CSVFile.csv', 'r') as csvFile:   # read mode
	csvReader = csv.reader(csvFile)
	count = 0
	for line in csvReader:
		if count == 0:
			print('Header: ' + str(line))
		else:
			print('Row: ' + str(line))
		count += 1
```

#### Writing into a CSV file:
```Python
with open('CSVFile.csv', 'a') as csvFile:   # append mode
	csvWriter = csv.writer(csvFile)
	csvWriter.writerow(['Column 1', 2, 'Column 3'])
```

#### Using Pandas

```Python
import pandas as pd

df = pd.read_csv('CSVFile.csv')   # Printing headers and indexing rows automatically (df: data frame)

df.loc[len(df.index)] = ['Column 1', 2, 'Column 3']


df.to_csv('CSVFile.csv', index = False)   # Writing a new line to the file
```



### Map, lambda, list comprehension {#map}

```python
l = [1, 2, 3, 4, 5]

def squares(x):
	return x*x

l_squares = list(map(squares, l))   # Wrap map object into a list


# Lambda function (anonymous function: lambda args:expression)
s = 'Hello World'
s = map(lambda x: x.upper(),s)   ## 'HELLO WORLD'; pass several arguments as list, e.g. [s1, s2]

# Other example
sum = lambda x,y: x+y
sum(4.5)   # 9

# List comprehension
l = [x for x in range(10) if x%2==0]   # [0, 2, 4, 6, 8]

l2 = [x for x in range(5)]
l2 = ['odd' if x%2!=0 else 'even' for x in l2]   # ['even', 'odd', 'even', 'odd', 'even']
````




## Important packages for Data Science {#important-packages-data-science}


### NumPy {#numpy}

- NumPy (Numerical Python, written in C): working with multidimensional arrays and matrices (linear algebra)
- Combined with SciPy and MathPlotlib
- Arrays: ndarray
- Important attributes of ndarray:

	```python
	ndarray.dim	  # Number of axes (dimensions) of the array
	ndarray.shape   # Tuple of integers giving the size of the array in each dimension; length of this shape tuple is the number of axes (ndim).
	ndarray.size   # Total number of elements of the array (= product of elements of the shape, e.g. m*n)
	ndarray.dtype   # Object describing the type of elements in the array (numpy.int16, numpy.int32, numpy.float64)
	ndarray.itemsize   # Size in bytes of each element of the array (e.g. float64: 64/8 = 8) => ndarray.dtype.itemsize
	ndarray.data   # Buffer containing the actual elements of the array (for accessing elements use indexing)

	np.array(24)   # zero-dimensional array (scalar)
	np.array([1,2,3,4])   # one-dimensional array
	np.array([1,1,1][1,2,1])   # two-dimensional array (second-order tensor, represents a matrix)
	npArr = np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])   # three-dimensional array (matrices as elements)
	npArr.shape   # (2, 2, 3)
	npArr.ndim   # 3

	# Creating an array of a specific number of dimensions
	np.array([1,2,3,4,5], ndim = 5)   # array([[[[[1, 2, 4, 4, 5]]]]])

	# Reshaping an array by changing its number of dimensions
	npArr = np.array([x for x in range(1,10)])
	npArr.reshape(3,3)   # two-dimensional array

	npArr = np.array([x for x in range(1,13)])
	npArr.reshape(2,2,3)   # three-dimensional array
	Or: npArr.reshape(3,2,2)   # Length of one-dimensional array is equal to the product of the three-dimensional array (12 = 3 * 2 * 2)

	# Dynamic converting of number of dimensions (for dynamic datasets)
	npArr.reshape(2,2,-1)   # Last dimension is automatically calculated into 3 (here)

	# Flatten an array to one dimension
	npArr.reshape(-1)   # array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
	```


#### Arithmetic operations with arrays {#arithmetic-operations-with-arrays}

```python
import numpy as np

A = np.array([1,2,1],[2,2,3])
B = np.array([3,4,5])

# Addition
np.add(A,B)

# Subtraction
np.subtract(B,A)

# Multiplication
np.multiply(A,B)

# Division
np.divide(A,B)

# Exponentiation
np.power(A,2)
np.power(A,B)
```


#### Conditional logic {#conditional-logic-in-arrays}

```python
import numpy as np

x = np.array([i for i in range(10)])   # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.where(x%2 == 0, 'Even', 'Odd')   # Conditions

# Condition list
condlist = [x<5, x>5]
choicelist = [x**2,x**3]
np.select(condlist,choicelist,default=x)   # array([0, 1, 4, 9, 16, 5, 216, 343, 512, 729])
```


#### Common mathematical and statistical functions {#common-functions}

```python
import numpy as np

arr = ([4,3,2],[10,1,0],[5,8,24])

# Minimum values
np.amin(arr)   # 0
np.amin(arr,axis=0)   # array([4, 1, 0]), vertical axis
np.amin(arr,axis=1)   # array([2, 0, 5]), horizontal axis

# Maximum values
np.max(arr)   # 24
np.amax(arr,axis=0)   # array([10, 8, 24]), vertical axis
np.amax(arr,axis=1)   # array([4, 10, 24]), horizontal axis

# Median, mean, standard deviation, variation, percentiles
np.median(arr)   # 4.0
np.mean(arr)   # 6.33333
np.std(arr)   # 6.9442222186
np.var(arr)   # 48.22222
np.percentile(arr,50)   # 4.0

# Mathematical functions
deg = np.array([0,30,45,60,90])
np.sin(deg*np.pi/180)
np.cos(deg*np.pi/180)
np.tan(deg*np.pi/180)

# Also: arcsin, arccos, arctan

# Floor and ceil functions
arr = np.array([0.1,0.8,-2.2,-9.87])
np.floor(arr)   # array([0., 0., -3, -10])
np.ceil(arr)   # array([1., 1., -2, -9])
```


#### Accessing and slicing arrays {#accessing-slicing-arrays}

```python
import numpy as np

array_1d = ([1,2,3,4,5,6])
array_2d = ([1,2,3],[4,5,6])
array_3d = ([[[1,2,3],[4,5,6]],[[7.8,9],[10,11,12]]])

array_1d[0]   # 1
array_1d[-3]   # 4

array_2d[1,2] or array_2d[1,-1]  # 6

array_3d[0,1,2]   # 6
array_3d[1,1,-1]   # 12

# Slicing
array_1d[3:5] or array_1d[-3:-1]  # array([4, 5])

array_2d[1,1:]   # array([5, 6])
array_2d[-2:-3:-1]   # array([1, 2, 3])

array_3d[0,1:,1:]   # array([5, 6])
```


#### Reading and writing files with Numpy {#reading-writing-numpy}

```python
import numpy as np

# Reading
arr = np.loadtxt('Sample.csv', delimiter=',', dtype=str)
arr = np.genfromtxt('Sample.csv', delimiter=',', dtype=str)

# Writing
arr = ([[1,2,3,],[4,5,6]])
np.savetxt('Sample_numpy.csv', arr, delimiter=',')
np.save('File.npy', arr)   # Needs less space
```



#### Simulations with Numpy {#simulations-numpy}

```python
import numpy as np
experiment = np.random.randint(0,2, size=10)   # [0 1 0 1 1 1 0 0 1 0]

coin_matrix = np.random.randint(0,2,size=(10000,10))   # 10000 rows with 10 coin tosses in each row

counts = coin_matrix.sum(axis=1)   # 1 = row-wise (default: column-wise)
counts.mean()   # average of heads in all the experiments
np.median(counts)   # center point in a sorted list
counts.min(), counts.max(), counts.std()

np.bincount(counts)   # array([  12,  104,  439, 1178, 2064, 2508, 2005, 1160,  424,  101,    5]) => Normal distribution

# Illustrate the distribution in percentages
unique_numbers = np.arange(0,11)
observed_times = np.bincount(counts)
for n, count in zip(unique_numbers, observed_times):
    print("{} heads observed {} times ({:0.1f}%)".format(n, count, 100*count/10000))
```

&nbsp;

**Simulation for returns on stock exchange**

```python
import matplotlib.pyplot as plt
%matplotlib inline

returns = np.random.normal(0.001, 0.02, 250)   # loc: mean (float); scale: standard deviation (float); size (int or tuple of ints, optional)

# Cumulative sum of the price
initial_price = 100
price = initial_price*np.exp(returns.cumsum())

# Plotting the graph
plt.plot(price)
plt.grid()
```






### Pandas {#pandas}

- Built on top of NumPy
- «Panel data» (Econometrics)
- Open-source library (BSD license): a simple and easy-to-use tool for data analysis
- Working with tabular data, time series data, matrix data

Advantages:
- Import and export data from JSON, CSV files etc.
- Easily handle missing values in a dataset
- Wrangle and manipulate data
- Merge multiple datasets
- Easily handle time series data

&nbsp;

**Two major types of data structures:**
- Series: one-dimensional ndarray with axis labels (n-dimensional array of NumPy library)
- Dataframes: two-dimensional heterogeneous tabular data similar to a spread-sheet, with data represented as rows and columns



#### Pandas Series {#pandas-series}

- Access the elements using the labels
- NumPy as underlying architecture
- Series are columns in Pandas dataframe



```python
import pandas as pd

# Create a list of five elements
l = [x for x in range(5)]

# Create a Series
s = pd.Series(l)

# Access an element
s[3]   # 3 (indices are assigned by default, from 0 to n-1)

# Add manual labels to the series (these indices may be duplicated)
s = pd.Series(l, index = ['a','b','c','d','e'])

s['e']   # 4


# Creating a series from a dictionary
data = {'a':1,'b':2,'c':3,'d':4}

s = pd.Series(data)

# Restrict the amount of data retrieved by indicating certain indices
s = pd.Series(data, index = ['a','b'])

# When giving indices not present in the series, the value corresponding to these indices is 'NaN'.
```


##### Querying a Series

```python
import pandas as pd

# Create a Series of eleven elements
s = pd.Series([x for x in range(1,11)])

# Accessing elements
s.iloc[0]   # 1
s.iat[8]   # 9

# Slicing a Series by specifying the indices
s[5:9]
s[-4:-1]

# Specifying conditions
s.where(s%2==0)   # If true, the value itself is returned. If false, a null value (NaN) is returned.

# Specify the value to be returned if the condition is false
s.where(s%2==0, 'Odd number')
s.where(s%2==0, s**2)   # Using a function if the condition is false

s.where(s%2==0, inplace=True)   # All odd values will be dropped from the Series
s.dropna()   # Drop all null values

# Fill a value in place of null values
s.fillna('Filled value')
```



#### Pandas Dataframe {#pandas-sataframe}

- Used to retabulate data

	```python
	import pandas as pd

	# Create an empty dataframe
	df = pd.DataFrame()

	type(df)   # pandas.core.frame.DataFrame

	# Reading a CSV file
	df = pd.read_csv('PandasExample.csv')

	# Printing the first five or last five members of the Dataframe
	df.head()
	df.tail()
	df.head(2)   # Print only the first two rows

	# Access elements by giving integer-based indices (0 to n-1)
	df.iloc[0]

	df.values   # Return the entire dataframe in a list of arrays

	# Reading the dataframe in chunks
	df = pd.read_csv('PandasExample.csv', chunksize=2)

	for chunk in df:
		print(chunk)   # Returns three separate dataframes (in this case)

	# Retrieving data by using conditions
	df = df[df['Age']>25]
	```



#### Pandas Panel {#pandas-panel}

- Multi-indexing in Pandas

	```python
	import pandas as pd

	df = pd.read_csv('HousePrices.csv', parse_dates = True)

	# Creating a multi-level index, using 'city' and 'date'
	df.set_index(['city','date'])   # Modify the original dataframe by the parameter: inplace = True

	df.index   # MultiIndex

	# Accessing a multiindex dataframe by the .loc method
	df.loc['Seattle']
	df.loc['Seattle'].loc['2014-05-02 00:00:00']
	```



#### Common functions in Pandas {#common-functions-in-pandas}

- General functions (e.g. Data manipulations, Top-level missing data, Top-level conversions)
- Input or output functions
- Series functions
- DataFrame functions
- Resampling
- Plotting


```python

## Input or output functions

# Reads an Excel spreadsheet into a DataFrame object
read_excel(io[SheetName, header, names, …])

# Writes into an Excel spreadsheet
DataFrame.to_excel(excel_writer[, …])

# Reads JSON data into a Pandas object
read_json([path_or_buf, orient, typ, dtype, …])

# Writes Pandas data into a JSON object
to_json(path_or_buf, obj[, orient, …])



## General functions

# Returns a reshaped DataFrame organized by index or column values
pivot(data[, index, columns, values])

# Detects missing values from an array-like object
isna(obj)

# Converts the argument to numeric
to_numeric(arg[, errors, downcast])



## Series attributes and functions

# Makes a copy of an object's indices and data
copy([deep])

# Provides integer-location-based indexing for selection by position
iloc()

# Returns addition of series and other element-wise addition
add(other[, level, fill_value, axis])
sub()   # Subtraction
mul()   # Multiplication



## DataFrame functions => attributes and methods for tabulated 2D data operations

.info
.values
.size



## Important methods in Pandas

# Replaces values where the condition is false
where(cond[, other, inplace, …])

# Iterates over DataFrame rows as (index, series) pairs
iterrows()

# Prints the top n rows
head([n])   # Default is 5
```



#### Statistical functions in Pandas {#statistical-functions-in-pandas}


**Note:** The skipna parameter is true by default in Mean, Median, Mode, STD, Skew (excluding NaN or null values from the result)


```python
# Returns the mean of the values over the requested axis
mean(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)

# Returns the median of the values over the requested axis
median(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)

# Retrieves the mode(s) of each element along the selected axis
mode(axis=0, numeric_only=False, dropna=True)

# Computes the sum of the values over the requested axis
sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs)

# Returns the standard deviation over the requested axis
std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)   # Normalized by n = -1 (change parameter ddof) ???

# Computes the pairwise covariance among the series (excluding the NaN or null values)
cov(min_periods=None, ddof=1)

# Computes the pairwise correlation of columns (excluding the NaN or null values)
cov(method='pearson', min_periods=1)   # other methods: kendall, spearman

# Returns an unbiased skew over the requested axis
skew(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)

# Returns unbiased kurtosis over the requested axis
kurt(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
kurtosis(array, axis=0, fisher=True, bias=True)


## Give a statistical snapshot of the dataframe (excluding the NaN or null values)
describe()
returns count()
mean()
std()
max()   # 25 %, 75 %
```

&nbsp;

**Pandas window functions under four categories**

- Rolling window functions
- Weighted window functions
- Expanding window functions
- Exponentially-weighted window functions

```python
count(), mean(), median() etc.
```



#### Date and Timedelta {#date-and-timedelta}

- Timeseries: a series of data points indexed in date or time order (occurring in successive equally spaced date or time intervals)

##### Pandas DatetimeIndex

```python
# Returns a fixed frequency DatetimeIndex object
date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)

# start and end: define the period and/or frequency of this object
# tz: timezone

## Attributes
- Day, date, month etc.
- Hour, minute, microsecond etc.
- Timetz, dayofweek, is_month_end etc. (utilities)

## Functions
strftime()   # Converts to index using specified date format

to_series([keep_tz, index, name])   # Creates a series with both index and values equal to index keys

to_frame([index, name])   # Creates a DateFrame with column containing the index
```


##### Timedelta

- This object represents a duration: the difference between two date or time values.

```python
Timedelta(value=<object object>, unit=None, **kwargs)
# day or date-related values: days or day
# time-related values: hour, hours, h, hr; minute, minutes, m, min; second, seconds, sec; microsecond, microseconds, micro, micros; millisecond, milliseconds, milli, millis; nanosecond, nanoseconds, nano, nanos, ns


## Attributes
- Days
- Seconds
- Microseconds


## Functions
isoformat()   # Formats a Timedelta object in ISO 8601 format

to_pytimedelta()   # Converts a Pandas Timedelta object into a Python timedelta object
```



#### IO tools {#io-tools}

- Some of the file format Pandas can handle: CSV, HTML, XML, JSON, LaTeX, SAS, SPSS, SQL, HDF5
- Specific read and write functions for every file format (e.g. read_html() and to_html())
- The name of the file is passed as a string argument to read and write functions.
- read_csv() and to_csv()
- Parameters:
	- Column and index locations and names
	- General parsing configurations
	- NA and missing data handling
	- Specifying column data types
	- etc.

```python
df = pd.read_csv('data.csv')
df.to_csv('output.csv')

# Df: Series, Tabulated Data, Dataframe, Multi-dimensional Data

df_json = pd.read_json('data.json')
df_json.to_json('output.json')
```




#### Categorical data {#categorical-data}

- Data that can be divided into different groups => data classification and categorization
- Examples: Blood type, Gender, Age, Customer rating
- dtype: category (in Series, Tabulated Data, and Dataframe)
- Categorical data is unordered.
- Categories are inferred from the data set.
- Categorical class, example:
  ```python
  grades = pd.Categorical(['A', 'A', 'C', 'F', 'B', 'D', 'B', 'C', 'F', 'D'], ordered=True)   # Create an ordered categorical

  # Reorder the categories
  grades = grades.reorder_categories(['F', 'D', 'C', 'B', 'A'])
  ```

A few supported operations:
- Renaming categories
- Appending new categories
- Setting new categories
- Removing unused categories
- Sorting and ordering categories



#### Working with text data (string data) {#working-with-text-data}

```python
text = pd.Series(['A', 'B', 'C', 'D', 'E', 'F', dtype='string'])   # Data read as object, string, or text type cannot be processed in Pandas.
```

- Data needs to be explicitly input as dtype='string', so it can be processed using regular string methods, e.g. str.lower(), str.upper(), str.len()

```python
split()   # parameter for splitting the text; list output accessed using get() or []
rsplit()   # reverse of the split() method
replace()   # Replace a part or the entire string with another string (pattern matching)
cat(sep=' ')   # Concatenate into a long string (using space as separator)
```



#### Iteration {#iteration}

- Pandas provides many ways of iterating two-dimensional and multidimensional data.
- Some iteration strategies:
	- Using the index: default index starting from 0; for loop
	- Using loc() method
	- Using iloc() method
	- Using iterrows() function
	- Using itertuples() function
	- Using apply() method

- Examples:

	```python
	# Using index for a dataframe
	for i in otcm.index:
		print(otcm['Problem'][i], otcm['Medecine'][i])

	# Using iloc():
	for i in range(len(otcm)):
		print(otcm.iloc[i, 0], otcm.iloc[i, 1], otcm.iloc[i, 2])

	# Using iterrows() (specifically made for dataframe):
	for index, row in otcm.iterrows():
		print(row['Problem'], row['Dosage'])

	# Using apply() that uses a lambda function
	print(otcm.apply(lambda row: row['Medecine'] + " -- is for --> " + str(row['Problem']), axis=1))
	```



#### Sorting in Pandas {#sorting-in-pandas}

```python
sort_values(by="")

otcm.sort_values(by='Problem')   # Sorting in alphabetical order (ascending=True by default)
```



#### Plotting with Pandas {#plotting-with-pandas}

```python
import pandas as pd

df = pd.read_csv('HousePrices.csv', parse_dates = True)

# Print the first five rows
df.head()

# Print the columns
df.columns

# Plot the graph
df.plot()   # Plots a line graph by default

df['price'].plot(legend=True)   # Plot a line graph for an individual column with legend

# Plot a scatter graph for two columns
df.plot(x='price', y='bedrooms', kind='scatter', legend=True)   # The legend is not displayed for this kind of graph.

# Plot the graph with a title and adjusted figure size
df.plot(title='Sample Plot', figsize=(15,10))

# Plot a box graph
df['price'].plot(kind='box', titel='Box Plot')

# Plot a histogram graph
df['price'].plot(kind='hist', titel='Histogram Plot')
```



### Data Analysis {#data-analysis}

- Possible definition (TechTarget): Data (sg./pl.) is information that has been translated into a form that is efficient for movement or processing with today's computers and transmission media. Data is information converted into binary digital form.
- Raw data: Data in its most basic digital format.
- Data in digital format is subjected to a definitive lifecyle and is amenable to process, analysis, visualization, and interpretation.
- Data lifecylce: also Information lifecycle, Information supply chain, Information resource lifecycle

- POSMAD (data lifecycle):
	- Plan: The phase that prepares for the resource.
	- Obtain: Allows acquiring the resource.
	- Store or share: Holds information about the resource and makes it accessible for use through some distribution method.
	- Maintain: Ensures the resource continues to work properly.
	- Apply: Uses the resource to support and address business needs.
	- Dispose: Delete or discard the resource when it is no longer useful.



#### Types of data {#types-of-data}

- Data:
	- Qualitative: not necessarily numerical, e.g. Colours, Sounds, Words, Symbols, Images, Videos
		- Nominal: names, labels; e.g. Gender: male, female; Results: pass, fail
		- Ordinal: numbers having natural, ordered categories; distances between categories are unknown; e.g. Service quality rating: Poor, Average, Good, Excellent, Outstanding
	- Quantitative: numbers, always associated with some units; e.g. employee count, speed of a car etc. => 'How many?', 'How often?'
		- Discrete: represent a whole number; e.g. population of a town in a particular year
		- Continous: data are represented on a continuum and indicate a precise number => floating point number with unit; e.g. volume of a tank (in cubic metre)



#### Working with Data {#working-with-data}

Data should be clean, precise, manipulatable, amenable for visualization.

- Data collection: from multiple sources, raw data need to be treated for analytics
- Data wrangling: format converting, weighing quality and context
	- Discovery
	- Structuring
	- Cleaning
	- Enriching
	- Validating
	- Publishing
- Data manipulation: missing values
	- Pivoting
	- Indexing
	- Applying functions
	- Separating
	- Merging
	- Sorting or ordering
- Data visualization: most-important aspect of working with data => cf. Matplotlib, Plotly, Seaborn (Python)




#### Importing and exporting data in Python {#importing-and-exporting-data-in-Python}

```python
df.to_csv('SampleFilegzip.gz', compression='gzip', index=False)   # Compress a file with gzip

df.read_csv('SampleFilegzip.gz')   # Read the compressed file
```



#### Regular expressions in Python {#regular-expressions-in-python}

```python
import re

s = 'This is a sample string.'

pattern = 'is'

re.search(pattern, s)   # (2, 4), first match
re.search(pattern, s).start()   # 2
re.search(pattern, s).end()   # 4
s[2:4]   # 'is'

pattern = '^This'

re.search(pattern, s)   # (0, 4)

re.findall('is', s)   # ['is']

# Load the regular expression into a variable
pattern = re.compile('\s\w+\s')
re.findall(pattern, s)   # [' is ', ' a ']

# Parse e-mail adresses
pattern = re.compile('\w+@\w+\..*...')

# Only top-level domains ???
pattern = re.compile('\w+@\w+\..*(...)')   # enclose the group in brackets

# Removing all non-letter characters, except spaces and points with white spaces
re.sub('[^A-Za-z0-9\s.]', ' ', s)

# Replace more than one white space
re.sub('[\s\s+', ' ', s)

# Find all upper-case words
upper_words = re.findall('[A-Z][A-Z]+', s)

# Replace the upper-case words with lower-case words using list comprehension
lower_words = [x.lower() for x in upper_words]

# And replace them in a text
for u, l in list(zip(upper_words, lower_words)):
	s = re.sub(u, l, s)
```



#### Accessing databases in Python {#accessing-databases-in-python}

```python
import sqlite3   # Just one example
import pandas as pd

df = read_csv('Sample.csv')

# Creating an sqlite3 connection
connection = sqlite3.connect('Test.db')

# Dump the dataframe into a database table
df.to_sql('Test', connection, if_exists='replace', index = False)   # Other options: append, raise (an error)

# Read data using cursor
cursor = connection.cursor()
query = "select * from Test"
cursor.execute(query)

results = cursor.fetchall()

# Reading data using pandas and sql-method
data = pd.read_sql(query, connection)

# Another example
query = 'select * from Test where city = "Seattle"'
data = pd.read_sql(query, connection)

# Updating or deleting rows inside a database using a library like pandas_sql ???
```




### Data Wrangling {#data-wrangling}


#### Pandorable Code (Idiomatic Pandas Code) {#pandorable-code}

```python
import pandas as pd

df = pd.read_csv('Example.csv')

# Access specific rows in the dataframe
df.iloc[0]
df.loc[10]


## Memory optimization

# Check the size and memory usage of a dataframe
df.size
df.memory_usage()   # deep=True

# Convert the datatype of a column
df['bedrooms'].astype('int32')

# Group data by certain columns (specify column and aggregation method)
df.groupby(['city']).mean()

# Or show the maximum value of a column for a specific column
df.groupby(['city']).max()


# Set an index in the original dataframe
df['date'] = df.to_datetime(df['date'])
df.set_index(['date'], inplace=True)

# Resample data
df.resample('D').mean()   # Or: 'M' (month), 'Y' (year)

# Plot a specific column
df['price'].plot()

# Chain methods to process certain aspects
df(pd.read_csv('Example.csv')
	.query('city == "Seattle"')
	.to_csv('Seattle.csv'))

# Display the count of different value categories in a column
data.groupby(['Mark']).size()
data.groupby(['Mark'])['Mark'].count()
```



#### Loading, Indexing, and Reindexing {#loading-indexing-and-reindexing}

```python
import pandas as pd

df = pd.read_csv('Example.csv')

# Set an index in the original dataframe
df.set_index(['date'], inplace=True)
df.set_index(['date', 'statezip'], inplace=True)   # Set multiple indices (in this order)

# Reset the index in the original dataframe
df.reset_index(inplace=True)

# Set the index when loading a file
df = pd.read_csv('Example.csv', index_col=['date'])
```




#### Merging dataframes {#merging-dataframes}

```python
import pandas as pd

df1 = pd.read_csv('Merge1.csv')
df2 = pd.read_csv('Merge2.csv')

# Merge two dataframes on the same column present in both of them (and having the same name)
pd.merge(df1, df2, on['date'])   # By default inner join is used. Other methods: outer, left, right

pd.merge(df1, df2, on['date'], how='outer')   # Missing field values filled up with 'NaN'

# Merge two dataframes on a column with different names
pd.merge(df1, df2, left_on['Date'], right_on=['date'])

# Merging on indexes
pd.merge(df1, df2, left_index=True, right_index=True)

# Merging on indexes using different suffixes (in a tuple)
pd.merge(df1, df2, left_index=True, right_index=True, suffixes=('_from_left', '_from_right'))


# Another merge method: df1.merge(df2)
```



#### Memory optimization in Python {#memory-optimization-in-python}

```python
import pandas as pd

df = pd.read_csv('Example.csv')

# Check the size and memory usage of a dataframe
df.size
df.memory_usage(deep=True)

df.dtypes

# Converting the dtype of a field
df['date'] = pd.to_datetime(df['date'])
df['street'] = df['street'].astype(str)

# Setting an index

# Reading data chunkwise for further optimization of memory usage
```



#### Data pre-processing {#data-pre-processing}

```python
import pandas as pd
import numpy as np

data = {'a': [1,2,np.nan], 'b': [1,np.nan,3], 'c': [1,np.nan,np.nan]}
df = pd.DataFrame(data)

# Checking the sum of null values
df.isnull().sum()

# Dropping the null values (rows having null values will get dropped)
df.dropna()   # inplace=True

# Dropping null values column-wise
df.dropna(axis=1)

# Dropping null values using a theshold (column-wise)
df.dropna(thresh=2, axis = 1)

# Dropping null values using a theshold (row-wise)
df.dropna(thresh=2)


# Fill null values
df.fillna(value=3)

# Forward-fill
df.fillna(method='ffill')   # null values replaced column-wise by the previous non-null values

# Backward-fill
df.fillna(method='bfill')   # null values replaced column-wise by the backward non-null values

# Replace null values based on mean value
df['a'].fillna(value=np.mean(df['a']))
```




#### Data binning: formatting and normalization {#data-binning-formatting-and-normalization}

Formatting: bringing data into a common format (used for further feature preparation)

Normalization: adjusting values measured in different scales to a common scale (in order to assign weights)
- Single-feature scaling
- Min-max scaling
- Z-score scaling
- Log scaling
- Clipping

```python
import pandas as pd
import numpy as np

df = pd.read_csv('CarPrices.csv', index_col=0)
df.dropna(inplace=True)

df.dtypes

# Put the data types into columns
cols = list(df.select_dtypes(np.object_).columns)

# Convert these columns into string type
df[cols] = df[cols].astype('string')

# Example for separating string values of a column
df['Company Name'] = df['CarName'].apply(lambda x: x.split()[0])   # Extract company names

# Check the unique values of a column
df['Company Name'].unique()

# Replacing wrong values in this column
cols = {'maxda': 'mazda', 'toyouta': 'toyota', 'vw': 'volkswagen'}


# Standardizing data
df['SF_peakrpm'] = df['peakrpm']/df['peakrpm'].max()

# Min-max scaling
df['MM_peakrpm'] = (df['peakrpm'] - df['peakrpm'].min()) / (df['peakrpm'].max() - df['peakrpm'].min())
```


```python
import pandas as pd
from scipy.stats import zscore

df = pd.read_csv('HousePrices.csv')

# Adding a column for z-score
df['zscore-price'] = zscore(df['price'])


# Detect outliers
import numpy as np

threshold = 2
df['outliers'] = np.where((df['zscore-price'] - threshold > 0), True, np.where((df['zscore-price'] + threshold < 0), True, False))

# Show statistical characteristics
df.describe()

# Drop the outliers from the dataset
df.drop((df[df['outliers'] == True]).index, inplace=True)
```



#### Describing data {#describing-data}

```python
import pandas as pd

df = pd.read_csv('HousePrices.csv')

df.head()
df.tail()

df.shape()  # (4600, 18) => rows and columns of the dataframe

df.columns

df.describe()   # Overview of the dataset (count [number of rows], mean, std, min, quartiles, max)
df.describe(percentiles=[0.1, 0.3, 0.5, 0.7, 0.9])

# Object types of the individual columns
df.dtypes
```




### Data Visualization {#cata-visualization}


#### Principles of information visualization {#principles-of-information-visualization}
- Data Visualization is the most important part of Data Science and Data Analytics.
- Edward Tufte's Information Principles:

**Graphical integrity**
1. Represent numbers so that they are directly proportional to the numerical quantities represented.
2. Defeat graphical distortion and ambiguity.
3. Display data variation, not design variation.
4. Use deflated and standardized units of monetary measurement in displaying time series.
5. Do not graphics quote data out of context.


**Data-ink**
- Show data above all else
- Maximize data-ink ratio
- Remove non-data-ink
- Remove redundant data-ink
- Revise and edit


**Chart junk**
- Excessive and unnecessary use of graphical effects
- Eliminate Moiré vibration, heavy grids, and self-promoting graphs (they showcase design skills rather than data).


**Data density**
- Use high-density graphs
- Graphs can be shrunk without compromising on legibility or information.


**Small multiples**
- Series of the same small graph frequented in one visual
- Small multiples are a great tool to visualize large quantities of data with a high number of dimensions.



#### Visualizing data using Pivot tables {#visualizing-data-using-pivot-tables}

Use a Pivot table to modify dataframes and view the data in a different way.

```python
import pandas as pd

df = pd.read_csv('HousePrices.csv')

# Pivot the dataset on 'city' using default aggregation function mean
pd.pivot_table(df, index = ['city'])

# Pivoting using multiple indices
pd.pivot_table(df, index = ['city', 'date'])

# Pivoting by changing the aggregation function
import numpy as np

pd.pivot_table(df, index = ['city', 'date'], aggfunc = np.max)

# Cf. the groupby function of Pandas
df.groupby(['city', 'date']).max()
```



#### Data Visualization library Mathplotlib {#data-visualization-library-mathplotlib}

- One of the earliest and most comprehensive libraries for static, animated, and interactive visualizations (created by John D. Hunter)

	```python
	import mathplotlib.pyplot as plt

	plot()   # Plotting two-dimensional graphs
	show()   # Display graphs
	```
- **Markers** of different varieties in the parameters of plot() method
- **Attribute linestyle** in the parameters of plot() method
- xlabel() and ylabel() for labelling the two dimensions of the graph
- grid() for adding grids to the graph

Graph types: e.g. ,  etc.

- scatter plots: scatter() => x and y data
- bar charts: bar() => two NumPy arrays, x-axis usually carries non-numerical data
- histograms: hist() => plot frequencies
- pie charts: pie() => NumPy array; input must add up to 100 %
- sub plots: more than one plot in a graphical area; have all the sub plots ready before calling the plot() method on the data, then call show() method.


```python
import pandas as pd
import mathplotlib.pyplot as plt

df = pd.read_csv('HousePrices.csv', parse_dates=['date'], index_col=['date'])
df.sort_index(inplace=True)

# Line plot
plt.plot(df['price'])

# Box plot
plt.boxplot(df['price'])

# Histogram
plt.hist(df['sqft_living'])

# Scatter plot
plt.xlabel('price')
plt.ylabel('sqft_living')
plt.titel('Scatter Plot')
plt.scatter(x=df['price'], y=df['sqft_living'])

# Bar plot
d = {'a': 10, 'b': 20, 'c': 13}
plt.bar(x = d.keys(), height = d.values())

# Pie chart
plt.pie(x = d.values(), labels = d.keys())

# Modify the features of a figure
plt.figure(figsize=(10,15), dpi=100)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Line Plot of Prices')
plt.legend()
plt.plot(df['price'])
plt.savefig('Prices.png')   # Save a plot


## 3D visualization

df = pd.read_csv('ADANIPORTS.csv', parse_dates=True)

df['H-L'] = df.High - df.Low
df['100MA'] = df['Close'].rolling(100).mean()

# Create the axis
ax = plt.axes(projection='3d')
ax = plt.scatter(df.index, df['H-L'], df['100MA'])   # 100 moving average
ax.set_xlabel('Index')
ax.set_ylabel('H-L')
ax.set_zlabel('100MA')

# Plot the graph
plt.show()
```

&nbsp;

**Mathplotlib and library geopandas**

```python
import geopandas as gpd

path = gpd.datasets.get_path('nybb')
df = gpd.read_file(path)

df.plot()

# Plot the Queens
df[df['BoroName']=='Queens'].plot()

# Show the area with a legend
df['area'] = df.area
df.plot('area', legend = True)

# Interact with the map in various ways (with overlays, zoom-in and zoom-out options, move)
df.explore()
```



#### Data Visualization library Seaborn {#data-visualization-library-seaborn}

- A high-level library for preparing statistical graphics (based on Mathplotlib)
- Closely integrated with Pandas data structures
- Comes with a number of example datasets.
- Supports a number of different dataset formats.
- Uses Pivot to convert long and wide form of data sets for data manipulation.
- Many functions automatically perform the necessary statistical estimation.
- Seaborn uses bootstrapping to compute confidence intervals and draw error bars (representing the uncertainty of the estimate).


```python
set_theme()   # Set a default theme
load_dataset()   # Load data

# Query and inspect data before plotting and visualizing them
head()
tail()

relplot()   # Plot graphs, parameters: **data**, x, y, hue, style, size etc.

implot()   # Enhance a scatter plot by including a linear regression model (and its uncertainty) => goes beyond descriptive statistics

displot()   # Plot and visualize statistical information automatically

catplot()   # For category plots

joinplot(), pairplot()   # Plot Composite views of multivariate datasets (visualize complex statistical data)
```

```python
import pandas as pd
import mathplotlib.pyplot as plt
import seaborn as sns
%mathplotlib inline

df = pd.read_csv('HousePrices.csv', index_col=['date'], parse_dates=True)
df.sort_index(inplace=True)

# Box plot
plt.figure(figsize=(7,7), dpi=100)
plt.title('Box Plot using Seaborn')
sns.boxplot(data = df, x = 'price')


## Statistical estimation and error bars (from Seaborn library's built-in fMRI data)

# FMRI data: dataframe with 1064 rows and 5 columns (subject, timepoint, event, region, and signal)
# Two different events: Stim, Cue; Two different regions: Parietal, Frontal

sns.set_theme()
fmri = sns.load_dataset('fmri')

sns.relplot(data=fmri, kind='line', x='timepoint', y='signal', col='region', hue='event', style='event')

# For both plots, time point (x-axis) and signal (y-axis) are the same.
# Interpretation:
# 	- Signals on the parietal are higher than those on the frontal
#	- The stim line shows that it peaks or troughs around the time point value of around 5.0.
#	- Similar inferences can be drawn on event cue.
#	- Both the signals are banded, so statistical estimation and error bars are superimposed on the signals line.
#	- The error bars are wider when the signal points are peaked or troughed. 



## Plotting 3D graphs

df = pd.read_csv('ADANIPORTS.csv', parse_dates=True)

df['H-L'] = df.High - df.Low
df['100MA'] = df['Close'].rolling(100).mean()

sns.set_style('darkgrid')

# Create the axis
ax = plt.axes(projection='3d')
ax = plt.scatter(df.index, df['H-L'], df['100MA'])   # 100 moving average
ax.set_xlabel('Index')
ax.set_ylabel('H-L')
ax.set_zlabel('100MA')

# Plot the graph
plt.show()


# Another 3D graph
import numpy as np

z1 = np.linspace(0,10,100)
x1 = np.cos(2*z1)
y1 = np.sin(2*z1)

sns.set_style('whitegrid')

ax = plt.axes(projection='3d')
ax.plot3D(x1,y1,z1)
plt.show()


# Plot 3D surfaces
def return_z(x,y):
	return 50-(x**2+y**2)   # Function needed to plot the meshgrid

sns.set_style('whitegrid')

# The data
x1,y1 = np.linspace(-5,5,50),np.linspace(-5,5,50)

x1,y1 = np.meshgrid(x1,y1)
z1 = return_z(x1,y1)

# Plot the 3D surface graph
ax = plt.axes(projection='3d')
ax.plot_surface(x1,y1,z1)
plt.show()
```




#### Data Visualization library Plotly {#data-visualization-library-plotly}

- Make interactive and publication-worthy graphs
- Line plots, Scatter plots, Bar and area charts, Error bars, Histograms, Heat maps, Box plots, Sub plots, Multiple axes plots, Bubble and polar charts
- Specializing in interactivity:
	- Zoom
	- Pan
	- Zoom-in
	- Zoom-out
	- Autoscale
	- Reset axis
- Download graphs in PNG format

&nbsp;

**Fundamental features:**

- Creating and updating figures
- Displaying figures
- Plotly express

- Basic charts: Scatter plots, Line Charts, Bar charts, Pie charts, Bubble plots
- Statistical charts: Error bars, Box Plots, Histograms, Distribution plots, Histogram plots
- Scientific charts: Contour plots, Heat maps, Ternary, Log plots
- Maps: Lines, Choropleths, Filled areas, Bubble Plots => directly on geographical maps
- Specialized plotting for artificial intelligence, machine learning, bioinformatics => generate e.g. regression or classification plots based an AI and ML datasets
- Plots like Vulcano plots, Manhattan plots, Clustergram (bioinformatics)
- 3D axes plots, 3D scatter plots, 3D surface plots, 3D sub plots, 3D camera control plots
- Subplot features: Mixed sub-plots, Map sub-plots, Table and chart sub-plots, Figure factory sub-plots
- Graphic-based transformations: filtering, invoke groupby, aggregate, multiple transformations
- Add custom control to graphs and figures, e.g. custom buttons, custom sliders, drop-down menus, range sliders, selectors
- Two- and three-dimensional animations in graphs => analyzing complex multi-variate data
- Generate dashboards


```python
# Changing the backend in Pandas options
import pandas as pd
pd.options.plotting.backend = 'plotly'

df = pd.read_csv('HousePrices.csv', index_col=['date'], parse_dates=True)
df.sort_index(inplace=True)

# Plot a line graph
df['price'].plot()   # Interaction: zoom-in, zoom-out, take screenshots, download png file etc.


# Using Plotly express directly
import plotly.express as px

fig = px.line(df, x=df.index, y='price', title='Line Plot using Plotly Express')
fig.show()

# Other plots
fig = px.histogram(df, x=df['price'])
fig.show()

fig = px.boxplot(df, x=df['price'], title='Box Plot')   # Shows quartile ranges, min, max etc.
fig.show()

# Show plot offered by Plotly by checking the directory
dir(px)

# Plot a 3D graph
fig = px.scatter_3d(df, x=df.index, y=df['price'], z=df['sqft_living'], title='3D graph using Plotly Express')   # Move the graph and visualize data in 3D
```



#### Data Visualization library Bokeh {#data-visualization-library-bokeh}

- Creating interactive visualizations for modern web browsers
- Build beautiful graphics, ranging from simple plots to complex dashboards with streaming datasets
- Create JavaScript-powered visualizations without writing any JavaScript
- Steps for building simple Bokeh graphs:
	1. Prepare data in Python; for simple graphs, use lists
	2. Call the figure() function; customize properties like title, tools, axes labels etc. using this function.
	3. Add renderers to the plot; e.g. use line() method to plot line plots (with attributes of the renderer, e.g. legend).
	4. Use show() or save() method.

- Glyphs: Bars, Lines, Hex tiles, Different polygon shaphes
- circle()   # Rendering circles (modify color and size)
- vbar()   # Rendering bars
- Include multiple renderers in the same plot to get customized graphs and rendering
- E.g. legend attributes: label_text_font, label_text_color, border_line_width, border_line_alpha
- For including annotations, import BoxAnnotation.
- Themes: Calibre, Dark_minimal, Night_sky, Contrast
- Enable responsive sizing by the attribute sizing_mode


```python
from bokeh.plotting import figure, show
from bokeh.io import output_notebook   # Plot the graphs inside the Jupyter notebook (by creating a new tab or cell with the graph)
import pandas as pd

output_notebook()   # Load BokehJS

df = pd.read_csv('HousePrices.csv', index_col=['date'], parse_dates=True)
df.sort_index(inplace=True)

# Line plot
graph = figure(title = 'Line Plot using Bokeh')
graph.line(df.index, df['price'])
show(graph)   # Zoom-in, zoom-out, move or save the graph etc.
```



### SciPy {#scipy}

- SciPy (Scientific Python) is built on top of NumPy (scientific computation library)
- Submodules for optimization, linear algebra, signal and image processing
- Basic data structure: multidimensional array (NumPy)
- Some features: constants, optimizers, graphs, special data, tests etc.

```python
from scipy import constants

dir(constants)   # list of all constants defined in SciPy

constants.pi   # Value of Pi

constants.degree   # Degree in radiants (=> np.pi/180)

np.sin(45*constants.degree)   # 0.70710678
```


### StatsModels {#statsmodels}

- Estimation of statistical models and perform statistical tests
- Important features: linear regression, ANOVA, time series analysis, statistical tests, graphics


```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset('Guerry', 'HistData').data   # An in-built data collection

data.head()   # Have a look a the data

# Build a model: lottery would be a function of literacy and the natural log of a population
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data = data).fit()

results.summary   # View the summary of the model (information needed to judge a model)


## Another example ##

# Import packages
import pandas as pd
import statsmodels.formula.api as smf

# Read CSV file
data = pd.read_csv('Advertising.csv', index_col = 0)

data.head()

# Build a model
model = smf.ols(formula = 'Sales ~ TV', data = data)   # Dependent variable: Sales

model = model.fit()

model.summary()

# View the parameters of the model
model.params

model.conf_int()
model.pvalues
model.rsquared


# Build a model on top of the other parameters
model = smf.ols(formula = 'Sales ~ TV + Radio + Newspaper', data = data).fit()

model.summary()	# R square has increased => better model performance

# Compare parameters in order to prove or refute null hypothesis etc.
```
