# Python Cheat Sheet for Data Analysts
## From Beginner to Advanced

---

## Table of Contents
1. [Basics & Fundamentals](#basics--fundamentals)
2. [Data Structures](#data-structures)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [File Operations](#file-operations)
6. [NumPy for Numerical Computing](#numpy-for-numerical-computing)
7. [Pandas for Data Analysis](#pandas-for-data-analysis)
8. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
9. [Data Visualization](#data-visualization)
10. [Statistical Analysis](#statistical-analysis)
11. [Advanced Pandas Techniques](#advanced-pandas-techniques)
12. [Performance Optimization](#performance-optimization)

---

## Basics & Fundamentals

### Variables and Data Types

```python
# Basic data types
integer_var = 42                    # int
float_var = 3.14159                 # float
string_var = "Hello, Data!"         # str
boolean_var = True                  # bool
none_var = None                     # NoneType

# Type checking and conversion
type(integer_var)                   # Returns: <class 'int'>
isinstance(float_var, float)        # Returns: True
str(integer_var)                    # Convert to string: "42"
int("100")                          # Convert to integer: 100
float("3.14")                       # Convert to float: 3.14
```

### String Operations

```python
# String manipulation
text = "data analysis"
text.upper()                        # "DATA ANALYSIS"
text.capitalize()                   # "Data analysis"
text.title()                        # "Data Analysis"
text.replace("data", "statistical") # "statistical analysis"

# String formatting (f-strings - most modern)
name = "Alice"
age = 30
f"My name is {name} and I'm {age} years old"

# Multiple lines
query = """
SELECT customer_id, total_sales
FROM sales_table
WHERE date >= '2024-01-01'
"""

# String methods useful for data cleaning
text = "  Data Science  "
text.strip()                        # "Data Science"
text.lower()                        # "  data science  "
text.split()                        # ["Data", "Science"]
"-".join(["2024", "01", "15"])      # "2024-01-15"

# Checking string content
"2024".isdigit()                    # True
"abc123".isalnum()                  # True
"Hello".startswith("He")            # True
"Python".endswith("on")             # True
```

### Operators

```python
# Arithmetic operators
10 + 5      # Addition: 15
10 - 5      # Subtraction: 5
10 * 5      # Multiplication: 50
10 / 5      # Division: 2.0
10 // 3     # Floor division: 3
10 % 3      # Modulus (remainder): 1
10 ** 2     # Exponentiation: 100

# Comparison operators
10 == 10    # Equal: True
10 != 5     # Not equal: True
10 > 5      # Greater than: True
10 < 5      # Less than: False
10 >= 10    # Greater or equal: True
10 <= 5     # Less or equal: False

# Logical operators
True and False      # False
True or False       # True
not True            # False

# Membership operators
5 in [1, 2, 3, 4, 5]        # True
"age" in {"name": "John"}   # False (checks keys in dict)
```

---

## Data Structures

### Lists (Ordered, Mutable)

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, True]
nested = [[1, 2], [3, 4], [5, 6]]

# Accessing elements
numbers[0]          # First element: 1
numbers[-1]         # Last element: 5
numbers[1:4]        # Slicing [start:end]: [2, 3, 4]
numbers[:3]         # First 3 elements: [1, 2, 3]
numbers[2:]         # From index 2 onwards: [3, 4, 5]
numbers[::2]        # Every 2nd element: [1, 3, 5]
numbers[::-1]       # Reverse: [5, 4, 3, 2, 1]

# Modifying lists
numbers.append(6)               # Add to end: [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)            # Insert at index: [0, 1, 2, 3, 4, 5, 6]
numbers.extend([7, 8])          # Extend list: [0, 1, 2, 3, 4, 5, 6, 7, 8]
numbers.remove(0)               # Remove first occurrence
numbers.pop()                   # Remove and return last element
numbers.pop(0)                  # Remove and return element at index
del numbers[0]                  # Delete element at index

# List methods
len(numbers)                    # Length
numbers.count(3)                # Count occurrences
numbers.index(4)                # Find index of value
numbers.sort()                  # Sort in place
numbers.reverse()               # Reverse in place
sorted(numbers)                 # Return sorted copy
numbers.clear()                 # Remove all elements

# List comprehension (powerful for data transformation)
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Tuples (Ordered, Immutable)

```python
# Creating tuples
coordinates = (10.5, 20.3)
single = (1,)                   # Note the comma for single element
unpacking = 1, 2, 3             # Parentheses optional

# Accessing elements (same as lists)
coordinates[0]                  # 10.5

# Tuple unpacking (very useful in data analysis)
x, y = coordinates
a, b, c = (1, 2, 3)

# Named tuples (useful for structured data)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(11, 22)
p.x                             # 11
p.y                             # 22
```

### Dictionaries (Key-Value Pairs, Mutable)

```python
# Creating dictionaries
person = {
    "name": "John Doe",
    "age": 30,
    "city": "New York"
}

# Alternative creation
person = dict(name="John Doe", age=30, city="New York")

# Accessing elements
person["name"]                  # "John Doe"
person.get("age")               # 30
person.get("salary", 0)         # Returns 0 if key doesn't exist

# Modifying dictionaries
person["age"] = 31              # Update value
person["country"] = "USA"       # Add new key-value pair
person.update({"age": 32, "job": "Analyst"})

# Removing elements
person.pop("city")              # Remove and return value
del person["country"]           # Delete key-value pair

# Dictionary methods
person.keys()                   # Get all keys
person.values()                 # Get all values
person.items()                  # Get all key-value pairs
"age" in person                 # Check if key exists

# Dictionary comprehension
squared = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Sets (Unordered, Unique Elements)

```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
unique_numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Set operations (useful for data deduplication)
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

set1.union(set2)                # {1, 2, 3, 4, 5, 6}
set1 | set2                     # Same as union
set1.intersection(set2)         # {3, 4}
set1 & set2                     # Same as intersection
set1.difference(set2)           # {1, 2}
set1 - set2                     # Same as difference
set1.symmetric_difference(set2) # {1, 2, 5, 6}

# Set methods
unique_numbers.add(6)           # Add element
unique_numbers.remove(1)        # Remove element (raises error if not found)
unique_numbers.discard(1)       # Remove element (no error if not found)
```

---

## Control Flow

### If-Elif-Else Statements

```python
# Basic if statement
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator (one-liner)
status = "Pass" if score >= 60 else "Fail"

# Multiple conditions
age = 25
has_license = True

if age >= 18 and has_license:
    print("Can drive")

# Checking for None or empty values
data = []
if data:  # False for empty list, None, 0, ""
    print("Data exists")
else:
    print("No data")
```

### Loops

```python
# For loops
numbers = [1, 2, 3, 4, 5]

for num in numbers:
    print(num)

# With index using enumerate (very useful!)
for index, value in enumerate(numbers):
    print(f"Index {index}: {value}")

# Loop through dictionary
person = {"name": "Alice", "age": 30, "city": "NYC"}

for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

# Range function
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):       # 2, 4, 6, 8 (start, stop, step)
    print(i)

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue                # Skip rest of loop iteration
    if i == 7:
        break                   # Exit loop completely
    print(i)

# Else clause with loops (rarely used but useful)
for i in range(5):
    if i == 10:
        break
else:
    print("Loop completed without break")
```

---

## Functions

### Basic Functions

```python
# Simple function
def greet(name):
    return f"Hello, {name}!"

result = greet("Alice")

# Multiple parameters with default values
def calculate_total(price, tax_rate=0.08, discount=0):
    subtotal = price - discount
    total = subtotal * (1 + tax_rate)
    return total

calculate_total(100)                    # Uses default tax_rate and discount
calculate_total(100, 0.10)              # Custom tax_rate
calculate_total(100, discount=20)       # Named argument

# Multiple return values
def get_statistics(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count if count > 0 else 0
    return total, count, average

total, count, avg = get_statistics([1, 2, 3, 4, 5])

# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Positional arguments: {args}")
    print(f"Keyword arguments: {kwargs}")

flexible_function(1, 2, 3, name="Alice", age=30)
```

### Lambda Functions (Anonymous Functions)

```python
# Simple lambda
square = lambda x: x**2
square(5)                       # 25

# Lambda with multiple arguments
multiply = lambda x, y: x * y
multiply(3, 4)                  # 12

# Commonly used with map, filter, sorted
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Sorting with lambda
data = [{"name": "Alice", "age": 30}, 
        {"name": "Bob", "age": 25}]
sorted_data = sorted(data, key=lambda x: x["age"])
```

### Advanced Function Concepts

```python
# Decorators (useful for timing, logging, etc.)
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def process_data(n):
    return sum(range(n))

process_data(1000000)

# Generator functions (memory efficient for large datasets)
def data_generator(n):
    for i in range(n):
        yield i ** 2

# Only generates values when needed
for value in data_generator(5):
    print(value)

# List comprehension vs generator expression
list_comp = [x**2 for x in range(1000)]     # Creates entire list in memory
gen_exp = (x**2 for x in range(1000))       # Creates values on demand
```

---

## File Operations

### Reading Files

```python
# Reading entire file
with open('data.txt', 'r') as file:
    content = file.read()

# Reading line by line (memory efficient)
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())

# Reading all lines into a list
with open('data.txt', 'r') as file:
    lines = file.readlines()

# CSV files (using csv module)
import csv

with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Skip header
    for row in csv_reader:
        print(row)

# CSV with DictReader (more convenient)
with open('data.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        print(row['column_name'])
```

### Writing Files

```python
# Writing to file
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("Second line\n")

# Appending to file
with open('output.txt', 'a') as file:
    file.write("Appended line\n")

# Writing CSV
import csv

data = [
    ['Name', 'Age', 'City'],
    ['Alice', 30, 'NYC'],
    ['Bob', 25, 'LA']
]

with open('output.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)

# Writing CSV from dictionaries
with open('output.csv', 'w', newline='') as file:
    fieldnames = ['Name', 'Age', 'City']
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerow({'Name': 'Alice', 'Age': 30, 'City': 'NYC'})
```

### JSON Operations

```python
import json

# Reading JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Writing JSON
data = {'name': 'Alice', 'age': 30, 'scores': [85, 90, 92]}

with open('output.json', 'w') as file:
    json.dump(data, file, indent=4)

# Converting between JSON strings and Python objects
json_string = json.dumps(data, indent=2)
python_obj = json.loads(json_string)
```

---

## NumPy for Numerical Computing

### Array Creation

```python
import numpy as np

# Creating arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Array creation functions
np.zeros((3, 4))                # 3x4 array of zeros
np.ones((2, 3))                 # 2x3 array of ones
np.full((2, 2), 7)              # 2x2 array filled with 7
np.eye(3)                       # 3x3 identity matrix
np.arange(0, 10, 2)             # Array from 0 to 10, step 2: [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)            # 5 evenly spaced values between 0 and 1
np.random.rand(3, 3)            # 3x3 array of random values [0, 1)
np.random.randn(3, 3)           # 3x3 array from standard normal distribution
np.random.randint(0, 10, (3, 3))# 3x3 array of random integers [0, 10)
```

### Array Attributes and Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape                       # (2, 3) - dimensions
arr.ndim                        # 2 - number of dimensions
arr.size                        # 6 - total number of elements
arr.dtype                       # dtype('int64') - data type
arr.itemsize                    # 8 - size of each element in bytes

# Reshaping
arr.reshape(3, 2)               # Change shape to 3x2
arr.flatten()                   # Flatten to 1D array
arr.T                           # Transpose
```

### Array Indexing and Slicing

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Basic indexing
arr[0, 1]                       # Element at row 0, column 1: 2
arr[1]                          # Entire row 1: [5, 6, 7, 8]
arr[:, 2]                       # Entire column 2: [3, 7, 11]

# Slicing
arr[0:2, 1:3]                   # Rows 0-1, columns 1-2
arr[:, ::2]                     # All rows, every 2nd column

# Boolean indexing (very powerful!)
mask = arr > 5
arr[mask]                       # [6, 7, 8, 9, 10, 11, 12]
arr[arr > 5]                    # Same as above

# Fancy indexing
arr[[0, 2], [1, 3]]             # Elements at (0,1) and (2,3)
```

### Array Operations

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Element-wise operations
arr1 + arr2                     # [6, 8, 10, 12]
arr1 - arr2                     # [-4, -4, -4, -4]
arr1 * arr2                     # [5, 12, 21, 32]
arr1 / arr2                     # [0.2, 0.33, 0.43, 0.5]
arr1 ** 2                       # [1, 4, 9, 16]

# Broadcasting (operations with different shapes)
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr + 10                        # Adds 10 to each element

# Statistical operations
arr = np.array([1, 2, 3, 4, 5])
np.mean(arr)                    # 3.0
np.median(arr)                  # 3.0
np.std(arr)                     # 1.414 (standard deviation)
np.var(arr)                     # 2.0 (variance)
np.min(arr)                     # 1
np.max(arr)                     # 5
np.sum(arr)                     # 15
np.prod(arr)                    # 120 (product)
np.cumsum(arr)                  # [1, 3, 6, 10, 15] (cumulative sum)

# Operations along axes
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.sum(arr2d, axis=0)           # [5, 7, 9] (sum along columns)
np.sum(arr2d, axis=1)           # [6, 15] (sum along rows)
np.mean(arr2d, axis=0)          # [2.5, 3.5, 4.5]
```

### Advanced NumPy Operations

```python
# Sorting
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
np.sort(arr)                    # [1, 1, 2, 3, 4, 5, 6, 9]
np.argsort(arr)                 # Indices that would sort the array

# Unique values
np.unique(arr)                  # [1, 2, 3, 4, 5, 6, 9]
np.unique(arr, return_counts=True)  # Also returns counts

# Where (conditional selection)
arr = np.array([1, 2, 3, 4, 5])
np.where(arr > 3, arr, 0)       # [0, 0, 0, 4, 5] (replace values ≤3 with 0)

# Concatenation
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
np.concatenate([arr1, arr2], axis=0)    # Stack vertically
np.vstack([arr1, arr2])                  # Same as above
np.hstack([arr1, arr2.T])                # Stack horizontally

# Splitting
arr = np.array([1, 2, 3, 4, 5, 6])
np.split(arr, 3)                # Split into 3 equal arrays

# Linear algebra
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.dot(a, b)                    # Matrix multiplication
a @ b                           # Same as np.dot
np.linalg.inv(a)                # Matrix inverse
np.linalg.det(a)                # Determinant
np.linalg.eig(a)                # Eigenvalues and eigenvectors
```

---

## Pandas for Data Analysis

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['NYC', 'LA', 'Chicago', 'Houston'],
    'Salary': [70000, 80000, 90000, 95000]
}
df = pd.DataFrame(data)

# From list of lists
data = [
    ['Alice', 25, 'NYC', 70000],
    ['Bob', 30, 'LA', 80000]
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'City', 'Salary'])

# From CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=';')                # Custom separator
df = pd.read_csv('data.csv', encoding='latin-1')    # Specific encoding
df = pd.read_csv('data.csv', nrows=1000)            # Read only first 1000 rows

# From Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# From SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)

# From JSON
df = pd.read_json('data.json')

# Creating Series (1D)
series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
```

### DataFrame Inspection

```python
df.head()                       # First 5 rows
df.head(10)                     # First 10 rows
df.tail()                       # Last 5 rows
df.shape                        # (rows, columns)
df.info()                       # Data types and non-null counts
df.describe()                   # Statistical summary of numerical columns
df.describe(include='all')      # Include non-numerical columns
df.columns                      # Column names
df.index                        # Row indices
df.dtypes                       # Data types of each column
df.memory_usage()               # Memory usage
df.nunique()                    # Number of unique values per column
df.value_counts('column_name')  # Count unique values in a column
```

### Selecting Data

```python
# Selecting columns
df['Name']                      # Single column (returns Series)
df[['Name', 'Age']]            # Multiple columns (returns DataFrame)

# Selecting rows by position
df.iloc[0]                      # First row
df.iloc[0:3]                    # First 3 rows
df.iloc[:, 0:2]                 # All rows, first 2 columns
df.iloc[0, 1]                   # Element at row 0, column 1

# Selecting rows by label
df.loc[0, 'Name']               # Element at row 0, column 'Name'
df.loc[0:2, ['Name', 'Age']]    # Rows 0-2, specific columns

# Boolean indexing (filtering)
df[df['Age'] > 30]              # Rows where Age > 30
df[df['City'] == 'NYC']         # Rows where City is NYC
df[(df['Age'] > 25) & (df['Salary'] > 75000)]  # Multiple conditions
df[df['Name'].isin(['Alice', 'Bob'])]          # Rows where Name in list

# Query method (alternative syntax)
df.query('Age > 30')
df.query('Age > 30 and Salary > 80000')
```

### Modifying DataFrames

```python
# Adding new columns
df['Bonus'] = df['Salary'] * 0.1
df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']

# Adding columns with conditions
df['Senior'] = df['Age'] > 35
df['Level'] = df['Age'].apply(lambda x: 'Senior' if x > 35 else 'Junior')

# Using assign (returns new DataFrame)
df_new = df.assign(
    Bonus=df['Salary'] * 0.1,
    Tax=df['Salary'] * 0.2
)

# Renaming columns
df.rename(columns={'Name': 'Employee_Name', 'Age': 'Employee_Age'}, inplace=True)

# Dropping columns
df.drop('Bonus', axis=1, inplace=True)
df.drop(['Bonus', 'Tax'], axis=1, inplace=True)

# Dropping rows
df.drop(0, axis=0, inplace=True)        # Drop row at index 0
df.drop([0, 1, 2], axis=0, inplace=True)

# Modifying values
df.loc[0, 'Age'] = 26                   # Change specific value
df.loc[df['City'] == 'NYC', 'Salary'] = 75000  # Conditional update

# Replacing values
df['City'].replace('NYC', 'New York', inplace=True)
df.replace({'NYC': 'New York', 'LA': 'Los Angeles'}, inplace=True)
```

### Sorting Data

```python
# Sort by single column
df.sort_values('Age')                           # Ascending (default)
df.sort_values('Age', ascending=False)          # Descending

# Sort by multiple columns
df.sort_values(['City', 'Age'], ascending=[True, False])

# Sort by index
df.sort_index()

# Sort with NA values at specific position
df.sort_values('Age', na_position='first')
```

### Grouping and Aggregation

```python
# Basic groupby
df.groupby('City')['Salary'].mean()
df.groupby('City')['Salary'].sum()

# Multiple aggregations
df.groupby('City').agg({
    'Salary': ['mean', 'sum', 'count'],
    'Age': ['min', 'max']
})

# Custom aggregation functions
df.groupby('City')['Salary'].agg(['mean', 'std', lambda x: x.max() - x.min()])

# Groupby multiple columns
df.groupby(['City', 'Department'])['Salary'].mean()

# Apply custom function to groups
def range_func(x):
    return x.max() - x.min()

df.groupby('City')['Salary'].apply(range_func)

# Transform (returns same shape as original)
df['Salary_normalized'] = df.groupby('City')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter groups
df.groupby('City').filter(lambda x: x['Salary'].mean() > 80000)
```

### Merging and Joining

```python
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 5, 6],
    'Salary': [70000, 80000, 90000, 95000]
})

# Inner join (only matching keys)
pd.merge(df1, df2, on='ID', how='inner')

# Left join (all from left, matching from right)
pd.merge(df1, df2, on='ID', how='left')

# Right join (all from right, matching from left)
pd.merge(df1, df2, on='ID', how='right')

# Outer join (all records from both)
pd.merge(df1, df2, on='ID', how='outer')

# Merge on multiple columns
pd.merge(df1, df2, on=['ID', 'Date'], how='inner')

# Merge with different column names
pd.merge(df1, df2, left_on='ID', right_on='Employee_ID', how='inner')

# Concatenate (stack DataFrames)
pd.concat([df1, df2], axis=0)           # Vertically (rows)
pd.concat([df1, df2], axis=1)           # Horizontally (columns)
pd.concat([df1, df2], ignore_index=True)# Reset index
```

### Pivot Tables

```python
# Basic pivot table
df.pivot_table(
    values='Salary',
    index='City',
    columns='Department',
    aggfunc='mean'
)

# Multiple aggregation functions
df.pivot_table(
    values='Salary',
    index='City',
    columns='Department',
    aggfunc=['mean', 'sum', 'count']
)

# Multiple values
df.pivot_table(
    values=['Salary', 'Age'],
    index='City',
    columns='Department',
    aggfunc='mean'
)

# With margins (totals)
df.pivot_table(
    values='Salary',
    index='City',
    columns='Department',
    aggfunc='mean',
    margins=True
)

# Pivot (simpler version without aggregation)
df.pivot(index='Date', columns='City', values='Temperature')
```

---

## Data Cleaning & Preprocessing

### Handling Missing Data

```python
# Detecting missing values
df.isnull()                     # Boolean DataFrame
df.isnull().sum()               # Count per column
df.isnull().sum() / len(df)     # Percentage missing
df[df.isnull().any(axis=1)]     # Rows with any missing values

# Filling missing values
df.fillna(0)                    # Fill with 0
df.fillna(df.mean())            # Fill with mean (numerical columns)
df.fillna(method='ffill')       # Forward fill (use previous value)
df.fillna(method='bfill')       # Backward fill (use next value)

# Fill with specific values per column
df.fillna({'Age': df['Age'].median(), 'City': 'Unknown'})

# Interpolate (for time series)
df['Value'].interpolate()
df['Value'].interpolate(method='linear')

# Dropping missing values
df.dropna()                     # Drop rows with any missing values
df.dropna(how='all')            # Drop rows where all values are missing
df.dropna(subset=['Age', 'City'])  # Drop if missing in specific columns
df.dropna(thresh=2)             # Drop rows with less than 2 non-NA values
df.dropna(axis=1)               # Drop columns with missing values
```

### Handling Duplicates

```python
# Detecting duplicates
df.duplicated()                 # Boolean Series
df.duplicated().sum()           # Count duplicates
df[df.duplicated()]             # Show duplicate rows

# Duplicates based on specific columns
df.duplicated(subset=['Name', 'City'])

# Removing duplicates
df.drop_duplicates()            # Keep first occurrence
df.drop_duplicates(keep='last') # Keep last occurrence
df.drop_duplicates(subset=['Name'])  # Based on specific column
```

### Data Type Conversions

```python
# Converting data types
df['Age'] = df['Age'].astype(int)
df['Price'] = df['Price'].astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df['Category'] = df['Category'].astype('category')

# Converting to string
df['ID'] = df['ID'].astype(str)

# Converting with errors handling
df['Numbers'] = pd.to_numeric(df['Numbers'], errors='coerce')  # Invalid → NaN
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
```

### String Operations

```python
# String methods (use .str accessor)
df['Name'].str.lower()                  # Lowercase
df['Name'].str.upper()                  # Uppercase
df['Name'].str.title()                  # Title case
df['Name'].str.strip()                  # Remove whitespace
df['Name'].str.replace('Dr.', 'Doctor')
df['Name'].str.contains('Alice')        # Boolean Series
df['Name'].str.startswith('A')
df['Name'].str.endswith('son')
df['Name'].str.len()                    # Length of each string

# Splitting strings
df['Full_Name'].str.split(' ', expand=True)  # Returns DataFrame with columns

# Extracting with regex
df['Phone'].str.extract(r'(\d{3})-(\d{4})')

# Example: Cleaning text data
df['Text'] = df['Text'].str.strip().str.lower().str.replace('[^a-z\s]', '', regex=True)
```

### Date and Time Operations

```python
# Creating datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Extracting components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0
df['DayName'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter
df['Week'] = df['Date'].dt.isocalendar().week

# Date arithmetic
df['NextWeek'] = df['Date'] + pd.Timedelta(days=7)
df['LastMonth'] = df['Date'] - pd.DateOffset(months=1)
df['DaysSince'] = (pd.Timestamp.now() - df['Date']).dt.days

# Date range
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
date_range = pd.date_range(start='2024-01-01', periods=12, freq='M')

# Resampling (for time series)
df.set_index('Date').resample('M').mean()  # Monthly average
df.set_index('Date').resample('W').sum()   # Weekly sum
```

### Outlier Detection and Handling

```python
# Z-score method
from scipy import stats
df['z_score'] = np.abs(stats.zscore(df['Value']))
outliers = df[df['z_score'] > 3]

# IQR method
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]

# Removing outliers
df_clean = df[(df['Value'] >= lower_bound) & (df['Value'] <= upper_bound)]

# Capping outliers (winsorizing)
df['Value_capped'] = df['Value'].clip(lower=lower_bound, upper=upper_bound)
```

---

## Data Visualization

### Matplotlib Basics

```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Line 1', color='blue', linewidth=2, marker='o')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Line Plot')
plt.legend()
plt.grid(True)
plt.show()

# Multiple plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x1, y1)
axes[0, 1].scatter(x2, y2)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data, bins=20)
plt.tight_layout()
plt.show()

# Scatter plot
plt.scatter(df['Age'], df['Salary'], alpha=0.5, c=df['Department'], cmap='viridis')
plt.colorbar(label='Department')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values, color='skyblue', edgecolor='navy')
plt.show()

# Histogram
plt.hist(df['Age'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot
plt.boxplot([df['Salary'], df['Bonus']], labels=['Salary', 'Bonus'])
plt.show()
```

### Seaborn (Statistical Visualization)

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')

# Distribution plot
sns.histplot(df['Age'], kde=True, bins=20)
plt.show()

# Box plot
sns.boxplot(x='Department', y='Salary', data=df)
plt.xticks(rotation=45)
plt.show()

# Violin plot (combines box plot and KDE)
sns.violinplot(x='Department', y='Salary', data=df)
plt.show()

# Scatter plot with regression line
sns.regplot(x='Age', y='Salary', data=df)
plt.show()

# Joint plot (scatter + histograms)
sns.jointplot(x='Age', y='Salary', data=df, kind='scatter')
plt.show()

# Pair plot (all combinations)
sns.pairplot(df[['Age', 'Salary', 'Bonus', 'Department']], hue='Department')
plt.show()

# Correlation heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.show()

# Count plot (categorical data)
sns.countplot(x='City', data=df, order=df['City'].value_counts().index)
plt.xticks(rotation=45)
plt.show()

# Bar plot with error bars
sns.barplot(x='Department', y='Salary', data=df, ci=95)
plt.show()
```

### Pandas Plotting

```python
# Line plot
df.plot(x='Date', y='Value', figsize=(10, 6))
plt.show()

# Multiple columns
df[['Sales', 'Profit']].plot(figsize=(10, 6))
plt.show()

# Bar plot
df.groupby('Department')['Salary'].mean().plot(kind='bar')
plt.show()

# Horizontal bar plot
df.groupby('City')['Salary'].mean().plot(kind='barh')
plt.show()

# Histogram
df['Age'].plot(kind='hist', bins=20)
plt.show()

# Box plot
df.boxplot(column='Salary', by='Department')
plt.show()

# Scatter plot
df.plot(kind='scatter', x='Age', y='Salary', alpha=0.5)
plt.show()

# Scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df[['Age', 'Salary', 'Bonus']], figsize=(12, 12))
plt.show()

# Area plot
df.groupby('Date')['Sales'].sum().plot(kind='area')
plt.show()

# Pie chart
df['Department'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
```

---

## Statistical Analysis

### Descriptive Statistics

```python
# Basic statistics
df['Salary'].mean()             # Average
df['Salary'].median()           # Median
df['Salary'].mode()             # Mode (most frequent)
df['Salary'].std()              # Standard deviation
df['Salary'].var()              # Variance
df['Salary'].min()              # Minimum
df['Salary'].max()              # Maximum
df['Salary'].sum()              # Sum
df['Salary'].quantile(0.25)     # 25th percentile (Q1)
df['Salary'].quantile([0.25, 0.5, 0.75])  # Multiple quantiles

# Skewness and kurtosis
df['Salary'].skew()             # Measure of asymmetry
df['Salary'].kurtosis()         # Measure of tailedness

# Summary statistics
df.describe()                   # For all numerical columns
df['Salary'].describe()         # For specific column
```

### Correlation Analysis

```python
# Correlation matrix
correlation_matrix = df.corr()
correlation_matrix = df.corr(method='pearson')   # Default
correlation_matrix = df.corr(method='spearman')  # Rank correlation
correlation_matrix = df.corr(method='kendall')   # Kendall's tau

# Correlation between two columns
df['Age'].corr(df['Salary'])

# Covariance
df.cov()
df['Age'].cov(df['Salary'])
```

### Hypothesis Testing

```python
from scipy import stats

# T-test (comparing means of two groups)
group1 = df[df['Department'] == 'Sales']['Salary']
group2 = df[df['Department'] == 'Marketing']['Salary']
t_statistic, p_value = stats.ttest_ind(group1, group2)

# Chi-square test (independence of categorical variables)
contingency_table = pd.crosstab(df['Department'], df['City'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# ANOVA (comparing means of multiple groups)
groups = [df[df['Department'] == dept]['Salary'] for dept in df['Department'].unique()]
f_statistic, p_value = stats.f_oneway(*groups)

# Normality test
statistic, p_value = stats.normaltest(df['Salary'])
statistic, p_value = stats.shapiro(df['Salary'])  # Shapiro-Wilk test
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data
X = df[['Age', 'Experience']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
```

---

## Advanced Pandas Techniques

### Window Functions (Rolling, Expanding, EWM)

```python
# Rolling window (moving average)
df['Sales_MA_7'] = df['Sales'].rolling(window=7).mean()
df['Sales_MA_30'] = df['Sales'].rolling(window=30).mean()

# Rolling with custom functions
df['Sales_Rolling_Max'] = df['Sales'].rolling(window=7).max()
df['Sales_Rolling_Std'] = df['Sales'].rolling(window=7).std()

# Expanding window (cumulative)
df['Cumulative_Sales'] = df['Sales'].expanding().sum()
df['Running_Average'] = df['Sales'].expanding().mean()

# Exponential weighted moving average
df['Sales_EWM'] = df['Sales'].ewm(span=7, adjust=False).mean()
```

### Advanced Groupby Operations

```python
# Multiple aggregations with custom names
df.groupby('Department').agg(
    avg_salary=('Salary', 'mean'),
    total_salary=('Salary', 'sum'),
    count=('Salary', 'count'),
    salary_range=('Salary', lambda x: x.max() - x.min())
)

# Groupby with multiple levels
df.groupby(['Department', 'City']).agg({'Salary': ['mean', 'count']})

# Named aggregations (Pandas 0.25+)
df.groupby('Department').agg(
    avg_salary=pd.NamedAgg(column='Salary', aggfunc='mean'),
    max_age=pd.NamedAgg(column='Age', aggfunc='max')
)

# Rank within groups
df['Salary_Rank'] = df.groupby('Department')['Salary'].rank(ascending=False)

# Cumulative sum within groups
df['Cumulative_Sales'] = df.groupby('Product')['Sales'].cumsum()

# Shift within groups (lag/lead)
df['Previous_Sales'] = df.groupby('Product')['Sales'].shift(1)
df['Next_Sales'] = df.groupby('Product')['Sales'].shift(-1)

# Percent change within groups
df['Sales_Change'] = df.groupby('Product')['Sales'].pct_change()
```

### MultiIndex (Hierarchical Indexing)

```python
# Creating MultiIndex
df_multi = df.set_index(['Department', 'City'])

# Accessing data in MultiIndex
df_multi.loc['Sales']                      # All rows in Sales department
df_multi.loc[('Sales', 'NYC')]             # Specific department and city
df_multi.loc[('Sales', 'NYC'), 'Salary']   # Specific value

# Cross-section
df_multi.xs('NYC', level='City')           # All rows where City is NYC

# Swapping levels
df_multi.swaplevel('Department', 'City')

# Sorting by index
df_multi.sort_index(level=0)

# Resetting index
df_flat = df_multi.reset_index()

# GroupBy with MultiIndex
df.groupby(['Department', 'City'])['Salary'].mean()

# Unstacking (pivot MultiIndex)
df_multi.unstack()
df_multi.unstack(level='City')

# Stacking (reverse of unstack)
df_stacked = df_wide.stack()
```

### Apply, Map, Applymap

```python
# Apply to columns (Series)
df['Salary_Thousands'] = df['Salary'].apply(lambda x: x / 1000)

# Apply to DataFrame (row-wise or column-wise)
df.apply(lambda x: x.max() - x.min())      # Column-wise (default, axis=0)
df.apply(lambda x: x.max() - x.min(), axis=1)  # Row-wise

# Apply with multiple columns
df['Total'] = df.apply(lambda row: row['Salary'] + row['Bonus'], axis=1)

# Map (for Series only, dictionary mapping)
df['City_Code'] = df['City'].map({'NYC': 1, 'LA': 2, 'Chicago': 3})

# Applymap (element-wise for entire DataFrame) - DEPRECATED, use map()
df_numeric = df[['Age', 'Salary']].applymap(lambda x: x * 2)  # Old way
df_numeric = df[['Age', 'Salary']].map(lambda x: x * 2)       # New way

# Using custom functions
def categorize_salary(salary):
    if salary < 50000:
        return 'Low'
    elif salary < 80000:
        return 'Medium'
    else:
        return 'High'

df['Salary_Category'] = df['Salary'].apply(categorize_salary)
```

### Categorical Data

```python
# Convert to categorical
df['Department'] = df['Department'].astype('category')

# Categorical benefits: reduced memory usage
df['Department'].memory_usage(deep=True)

# Ordered categories
df['Education'] = pd.Categorical(
    df['Education'],
    categories=['High School', 'Bachelor', 'Master', 'PhD'],
    ordered=True
)

# Add new categories
df['Department'] = df['Department'].cat.add_categories(['New_Dept'])

# Remove unused categories
df['Department'] = df['Department'].cat.remove_unused_categories()

# Rename categories
df['Department'] = df['Department'].cat.rename_categories({
    'Sales': 'Sales_Dept',
    'Marketing': 'Marketing_Dept'
})

# Get dummy variables (one-hot encoding)
pd.get_dummies(df['Department'])
pd.get_dummies(df, columns=['Department', 'City'])
```

### Advanced Merging Techniques

```python
# Merge with indicator (shows merge source)
merged = pd.merge(df1, df2, on='ID', how='outer', indicator=True)

# Merge with suffixes
merged = pd.merge(df1, df2, on='ID', suffixes=('_left', '_right'))

# Merge on index
df1.merge(df2, left_index=True, right_index=True)

# Merge with validation
pd.merge(df1, df2, on='ID', validate='one_to_one')   # Ensures no duplicates
pd.merge(df1, df2, on='ID', validate='one_to_many')

# Join (simpler merge on index)
df1.join(df2, how='left')
df1.join(df2, lsuffix='_left', rsuffix='_right')
```

### Time Series Analysis

```python
# Set datetime index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# Resampling
df.resample('D').mean()         # Daily average
df.resample('W').sum()          # Weekly sum
df.resample('M').last()         # Last value of each month
df.resample('Q').median()       # Quarterly median

# Rolling window operations
df['MA_7'] = df['Sales'].rolling(window=7).mean()
df['MA_30'] = df['Sales'].rolling(window=30).mean()

# Shift (lag/lead)
df['Sales_Yesterday'] = df['Sales'].shift(1)
df['Sales_Tomorrow'] = df['Sales'].shift(-1)

# Percent change
df['Sales_Change'] = df['Sales'].pct_change()
df['Sales_Change_7D'] = df['Sales'].pct_change(periods=7)

# Differencing
df['Sales_Diff'] = df['Sales'].diff()

# Time-based selection
df['2024']                      # All data from 2024
df['2024-01']                   # All data from January 2024
df['2024-01-01':'2024-01-31']   # Specific date range

# Business day frequency
business_days = pd.bdate_range(start='2024-01-01', end='2024-12-31')
```

---

## Performance Optimization

### Memory Optimization

```python
# Check memory usage
df.memory_usage(deep=True)
df.info(memory_usage='deep')

# Optimize data types
# Before optimization
df['ID'] = df['ID'].astype('int64')         # 8 bytes per value

# After optimization
df['ID'] = df['ID'].astype('int32')         # 4 bytes per value
df['ID'] = df['ID'].astype('int16')         # 2 bytes per value (if range allows)

# Optimize categorical columns
df['Category'] = df['Category'].astype('category')

# Function to optimize DataFrame
def optimize_dtypes(df):
    # Optimize integers
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize floats
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize objects to category (if unique values < 50% of total)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    return df

df_optimized = optimize_dtypes(df)
```

### Efficient Data Processing

```python
# Use vectorization instead of loops
# Bad (slow)
result = []
for value in df['Value']:
    result.append(value * 2)
df['Result'] = result

# Good (fast)
df['Result'] = df['Value'] * 2

# Use .values or .to_numpy() for NumPy operations
# Good for numerical operations
result = df['Value'].values * 2
result = df['Value'].to_numpy() * 2

# Use eval() for complex expressions
# Can be faster for large DataFrames
df.eval('Result = Value1 + Value2 * Value3', inplace=True)

# Use query() instead of boolean indexing
# Bad (slower for large DataFrames)
df[(df['Age'] > 25) & (df['City'] == 'NYC')]

# Good (faster)
df.query('Age > 25 and City == "NYC"')

# Chunking for large files
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed_chunk = chunk[chunk['Value'] > 100]
    chunks.append(processed_chunk)

df = pd.concat(chunks, ignore_index=True)

# Use appropriate data structures
# For frequent lookups: use dictionaries or sets
# For append operations: use lists, then convert to DataFrame
```

### Parallel Processing

```python
# Using multiprocessing
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    # Your processing logic
    chunk['Result'] = chunk['Value'] * 2
    return chunk

# Split DataFrame
chunks = np.array_split(df, 4)  # Split into 4 chunks

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_chunk, chunks)

# Combine results
df_result = pd.concat(results)

# Using joblib (easier interface)
from joblib import Parallel, delayed

def process_row(row):
    # Your processing logic
    return row['Value'] * 2

results = Parallel(n_jobs=4)(delayed(process_row)(row) for _, row in df.iterrows())
df['Result'] = results

# Using Dask for out-of-core computation (very large datasets)
import dask.dataframe as dd

# Read large CSV with Dask
ddf = dd.read_csv('large_file.csv')

# Operations are lazy (not executed immediately)
result = ddf[ddf['Value'] > 100].groupby('Category')['Value'].mean()

# Compute to get actual result
result.compute()
```

### Best Practices

```python
# 1. Use inplace=True sparingly (can be slower)
# Bad
df.drop('Column', axis=1, inplace=True)

# Good
df = df.drop('Column', axis=1)

# 2. Chain operations efficiently
result = (df
    .query('Age > 25')
    .groupby('City')['Salary']
    .mean()
    .sort_values(ascending=False)
)

# 3. Use .loc for setting values (avoid chained assignment)
# Bad
df[df['Age'] > 30]['Salary'] = 100000  # SettingWithCopyWarning

# Good
df.loc[df['Age'] > 30, 'Salary'] = 100000

# 4. Profile your code
import time

start = time.time()
# Your code here
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

# Or use %%timeit in Jupyter
# %%timeit
# df['Result'] = df['Value'] * 2

# 5. Use itertuples() instead of iterrows() when needed
# Bad (very slow)
for index, row in df.iterrows():
    print(row['Name'])

# Better (faster)
for row in df.itertuples():
    print(row.Name)

# Best (avoid iteration when possible)
print(df['Name'].tolist())
```

---

## Additional Resources and Tips

### Common Patterns for Data Analysts

```python
# 1. Reading and initial exploration
df = pd.read_csv('data.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.head())

# 2. Data cleaning pipeline
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        'Age': df['Age'].median(),
        'City': 'Unknown',
        'Salary': df['Salary'].mean()
    })
    
    # Convert data types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Category'] = df['Category'].astype('category')
    
    # Remove outliers
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Salary'] >= Q1 - 1.5 * IQR) & (df['Salary'] <= Q3 + 1.5 * IQR)]
    
    return df

df_clean = clean_data(df)

# 3. Creating analysis report
def create_summary_report(df):
    report = {
        'total_records': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'numerical_summary': df.describe(),
        'categorical_summary': df.describe(include='object'),
        'missing_values': df.isnull().sum(),
        'duplicates': df.duplicated().sum()
    }
    return report

# 4. Exporting results
# To CSV
df.to_csv('output.csv', index=False)

# To Excel with multiple sheets
with pd.ExcelWriter('analysis.xlsx') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    df.describe().to_excel(writer, sheet_name='Summary')
    df.groupby('Category')['Value'].mean().to_excel(writer, sheet_name='Grouped')

# To JSON
df.to_json('output.json', orient='records', indent=2)
```

### Useful Keyboard Shortcuts (Jupyter/IPython)

```
Ctrl + Enter: Run current cell
Shift + Enter: Run current cell and move to next
Alt + Enter: Run current cell and insert new cell below
Tab: Autocomplete
Shift + Tab: Show documentation
Ctrl + /: Comment/uncomment
Esc + M: Convert cell to Markdown
Esc + Y: Convert cell to Code
```

### Quick Reference - Data Types

```python
# Python built-in types
int       # Integer: 1, 2, 3
float     # Floating point: 1.0, 2.5
str       # String: "Hello"
bool      # Boolean: True, False
list      # List: [1, 2, 3]
tuple     # Tuple: (1, 2, 3)
dict      # Dictionary: {"key": "value"}
set       # Set: {1, 2, 3}

# NumPy types
np.int8, np.int16, np.int32, np.int64
np.float16, np.float32, np.float64
np.bool_
np.object_

# Pandas types
object        # Text or mixed
int64         # Integer
float64       # Floating point
bool          # Boolean
datetime64    # Date and time
timedelta     # Time difference
category      # Categorical
```

---

## End of Cheat Sheet

This comprehensive guide covers Python from basic syntax to advanced data analysis techniques specifically tailored for data analysts. Keep this handy as a reference while working on your projects!

**Remember**: 
- Practice is key to mastery
- Vectorization is almost always faster than loops
- Profile your code to find bottlenecks
- Read the documentation when stuck
- Join communities (Stack Overflow, Reddit r/Python, r/datascience)

Happy analyzing! 📊🐍
