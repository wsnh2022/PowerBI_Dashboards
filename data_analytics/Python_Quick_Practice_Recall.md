# VARIABLES
x = 5                    # assignment
x, y = 1, 2             # multiple assignment
x, y = y, x             # swap values

# DATA TYPES
# Numbers
int_num = 42            # integer
float_num = 3.14        # float
complex_num = 2 + 3j    # complex

# Strings
s = "hello"             # or 'hello'
s = """multi
line"""                 # triple quotes
f"value: {x}"           # f-string formatting

# Boolean
is_true = True          # or False

# Collections
lst = [1, 2, 3]         # list (mutable)
tpl = (1, 2, 3)         # tuple (immutable)
st = {1, 2, 3}          # set (unique, unordered)
dct = {"a": 1, "b": 2}  # dictionary (key-value)

# BASIC OPERATIONS
# Arithmetic
+ - * /                 # add, subtract, multiply, divide
//                      # floor division (5 // 2 = 2)
%                       # modulo (5 % 2 = 1)
**                      # exponent (2 ** 3 = 8)

# Comparison
== != < > <= >=         # equal, not equal, less, greater

# Logical
and or not              # logical operators

# String operations
"hi" + "!" = "hi!"      # concatenation
"hi" * 3 = "hihihi"     # repetition
"h" in "hi" = True      # membership
len("hi") = 2           # length
s[0] s[-1] s[1:3]       # indexing, slicing

# List operations
lst.append(4)           # add to end
lst.pop()               # remove last
lst[0] = 99             # modify by index
len(lst)                # length

# Dict operations
dct["key"]              # access value
dct["new"] = 3          # add/update
dct.get("key", default) # safe access
dct.keys() .values()    # get keys/values

# Type conversion
int("5") str(5)         # convert types
float("3.14")           # string to float
list("abc") = ['a','b','c']

# CONTROL FLOW
# If-elif-else
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")

# Ternary
result = "yes" if x > 5 else "no"

# LOOPS
# For loop
for i in range(5):          # 0,1,2,3,4
    print(i)

for item in lst:            # iterate list
    print(item)

for k, v in dct.items():    # iterate dict
    print(k, v)

# While loop
while x < 10:
    x += 1
    
# Loop controls
break       # exit loop
continue    # skip to next iteration

# LIST COMPREHENSIONS
[x**2 for x in range(5)]                    # [0,1,4,9,16]
[x for x in range(10) if x % 2 == 0]        # evens only
{x: x**2 for x in range(5)}                 # dict comprehension

# FUNCTIONS
def func(param1, param2=default):           # default arg
    """docstring"""
    return value

lambda x: x**2                              # anonymous function

# SLICING [start:stop:step]
lst[1:4]        # items 1 to 3
lst[:3]         # first 3
lst[3:]         # from index 3 to end
lst[-2:]        # last 2
lst[::2]        # every 2nd item
lst[::-1]       # reverse

# UNPACKING
a, b, c = [1, 2, 3]
a, *rest = [1, 2, 3, 4]     # a=1, rest=[2,3,4]
*start, last = [1, 2, 3]    # start=[1,2], last=3

# COMMON BUILT-INS
len() sum() min() max()     # aggregate functions
sorted(lst)                 # return sorted copy
enumerate(lst)              # (index, value) pairs
zip(lst1, lst2)             # pair up iterables
map(func, lst)              # apply function
filter(func, lst)           # filter by condition
any(lst) all(lst)           # boolean checks

# STRING METHODS
s.upper() .lower()          # case conversion
s.strip()                   # remove whitespace
s.split(",")                # split to list
",".join(lst)               # join list to string
s.replace("old", "new")     # replace substring
s.startswith() .endswith()  # check prefix/suffix

# ERROR HANDLING
try:
    risky_operation()
except ValueError:
    handle_error()
except Exception as e:
    print(e)
finally:
    cleanup()

# FILE I/O
with open("file.txt", "r") as f:    # r, w, a modes
    content = f.read()              # read all
    lines = f.readlines()           # list of lines
    
# USEFUL PATTERNS
# Check existence
if key in dct:
if item in lst:
if not lst:             # check if empty

# Default dict value
value = dct.get("key", 0)

# Chaining comparisons
if 0 < x < 10:

# Multiple conditions
if x in [1, 2, 3]:
