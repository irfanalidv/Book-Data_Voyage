#!/usr/bin/env python3
"""
Chapter 2: Python for Data Science
Data Voyage: Building the Foundation for Data Analysis

This script covers Python fundamentals essential for data science with
actual code execution and output examples.
"""

import sys
import platform
import time
from datetime import datetime
import math
import random


def main():
    print("=" * 80)
    print("CHAPTER 2: PYTHON FOR DATA SCIENCE")
    print("=" * 80)
    print()

    # Section 2.1: Python Basics
    print("2.1 PYTHON BASICS")
    print("-" * 40)
    demonstrate_python_basics()

    # Section 2.2: Data Structures
    print("\n2.2 DATA STRUCTURES")
    print("-" * 40)
    demonstrate_data_structures()

    # Section 2.3: Control Flow
    print("\n2.3 CONTROL FLOW")
    print("-" * 40)
    demonstrate_control_flow()

    # Section 2.4: Functions and OOP
    print("\n2.4 FUNCTIONS AND OBJECT-ORIENTED PROGRAMMING")
    print("-" * 40)
    demonstrate_functions_and_oop()

    # Section 2.5: File I/O and Error Handling
    print("\n2.5 FILE I/O AND ERROR HANDLING")
    print("-" * 40)
    demonstrate_file_io_and_error_handling()

    # Section 2.6: Python Packages for Data Science
    print("\n2.6 PYTHON PACKAGES FOR DATA SCIENCE")
    print("-" * 40)
    demonstrate_data_science_packages()

    # Chapter Summary
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Python fundamentals - Variables, data types, and basic operations")
    print("✅ Data structures - Lists, dictionaries, sets, and tuples")
    print("✅ Control flow - Conditionals, loops, and comprehensions")
    print("✅ Functions and OOP - Modular code and object-oriented design")
    print("✅ File I/O and error handling - Working with files and exceptions")
    print("✅ Data science packages - Essential libraries for analysis")
    print()
    print("Next: Chapter 3 - Mathematics and Statistics")
    print("=" * 80)


def demonstrate_python_basics():
    """Demonstrate basic Python concepts."""
    print("Python Environment Information:")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Variables and Data Types:")
    print("-" * 30)

    # Numeric types
    integer_num = 42
    float_num = 3.14159
    complex_num = 2 + 3j

    print(f"Integer: {integer_num} (type: {type(integer_num)})")
    print(f"Float: {float_num} (type: {type(float_num)})")
    print(f"Complex: {complex_num} (type: {type(complex_num)})")
    print()

    # String operations
    text = "Data Science"
    print(f"Original string: '{text}'")
    print(f"Length: {len(text)}")
    print(f"Uppercase: '{text.upper()}'")
    print(f"Lowercase: '{text.lower()}'")
    print(f"Split: {text.split()}")
    print(f"Replace 'Science' with 'Analysis': '{text.replace('Science', 'Analysis')}'")
    print()

    # Boolean operations
    is_true = True
    is_false = False
    print(f"Boolean values: {is_true}, {is_false}")
    print(f"Logical AND: {is_true and is_false}")
    print(f"Logical OR: {is_true or is_false}")
    print(f"Logical NOT: {not is_true}")
    print()

    # Basic arithmetic
    a, b = 15, 4
    print(f"Arithmetic operations with {a} and {b}:")
    print(f"  Addition: {a} + {b} = {a + b}")
    print(f"  Subtraction: {a} - {b} = {a - b}")
    print(f"  Multiplication: {a} * {b} = {a * b}")
    print(f"  Division: {a} / {b} = {a / b}")
    print(f"  Floor division: {a} // {b} = {a // b}")
    print(f"  Modulo: {a} % {b} = {a % b}")
    print(f"  Exponentiation: {a} ** {b} = {a ** b}")
    print()

    # String formatting
    name = "Alice"
    age = 30
    salary = 75000.50
    print("String formatting examples:")
    print(f"  f-string: {name} is {age} years old and earns ${salary:,.2f}")
    print(
        "  .format(): {} is {} years old and earns ${:,.2f}".format(name, age, salary)
    )
    print("  %% operator: %s is %d years old and earns $%.2f" % (name, age, salary))
    print()


def demonstrate_data_structures():
    """Demonstrate Python data structures."""
    print("Lists - Mutable sequences:")
    print("-" * 30)

    # Lists
    numbers = [1, 2, 3, 4, 5]
    mixed_list = [1, "hello", 3.14, True, [1, 2, 3]]

    print(f"Numbers list: {numbers}")
    print(f"Mixed list: {mixed_list}")
    print(f"Length: {len(numbers)}")
    print(f"First element: {numbers[0]}")
    print(f"Last element: {numbers[-1]}")
    print(f"Slicing [1:3]: {numbers[1:3]}")
    print(f"Slicing [::2]: {numbers[::2]}")

    # List operations
    numbers.append(6)
    numbers.insert(0, 0)
    numbers.extend([7, 8])
    print(f"After operations: {numbers}")
    print(f"Sorted: {sorted(numbers)}")
    print(f"Reversed: {list(reversed(numbers))}")
    print()

    print("Dictionaries - Key-value pairs:")
    print("-" * 30)

    # Dictionaries
    person = {
        "name": "John Doe",
        "age": 28,
        "city": "New York",
        "skills": ["Python", "SQL", "Machine Learning"],
    }

    print(f"Person dictionary: {person}")
    print(f"Name: {person['name']}")
    print(f"Skills: {person['skills']}")
    print(f"Keys: {list(person.keys())}")
    print(f"Values: {list(person.values())}")
    print(f"Items: {list(person.items())}")

    # Dictionary operations
    person["experience"] = 5
    person["skills"].append("Deep Learning")
    print(f"After updates: {person}")
    print()

    print("Sets - Unique unordered collections:")
    print("-" * 30)

    # Sets
    unique_numbers = {1, 2, 2, 3, 3, 4, 5, 5}
    print(f"Set from list with duplicates: {unique_numbers}")

    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}

    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
    print(f"Union: {set1 | set2}")
    print(f"Intersection: {set1 & set2}")
    print(f"Difference: {set1 - set2}")
    print(f"Symmetric difference: {set1 ^ set2}")
    print()

    print("Tuples - Immutable sequences:")
    print("-" * 30)

    # Tuples
    coordinates = (10, 20)
    rgb_color = (255, 128, 0)

    print(f"Coordinates: {coordinates}")
    print(f"RGB color: {rgb_color}")
    print(f"X coordinate: {coordinates[0]}")
    print(f"Y coordinate: {coordinates[1]}")
    print(f"Tuple unpacking: x={coordinates[0]}, y={coordinates[1]}")
    print()


def demonstrate_control_flow():
    """Demonstrate Python control flow structures."""
    print("Conditional Statements:")
    print("-" * 30)

    # If-elif-else
    score = 85
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    print(f"Score: {score}, Grade: {grade}")

    # Ternary operator
    status = "Pass" if score >= 60 else "Fail"
    print(f"Status: {status}")
    print()

    print("Loops:")
    print("-" * 30)

    # For loop with range
    print("Counting from 1 to 5:")
    for i in range(1, 6):
        print(f"  {i}", end=" ")
    print()

    # For loop with enumerate
    fruits = ["apple", "banana", "cherry"]
    print("Fruits with index:")
    for index, fruit in enumerate(fruits):
        print(f"  {index}: {fruit}")

    # For loop with dictionary
    print("Person details:")
    person = {"name": "Alice", "age": 30, "city": "Boston"}
    for key, value in person.items():
        print(f"  {key}: {value}")

    # While loop
    print("Countdown:")
    count = 5
    while count > 0:
        print(f"  {count}...", end=" ")
        count -= 1
        time.sleep(0.5)
    print("Blast off!")
    print()

    print("List Comprehensions:")
    print("-" * 30)

    # Basic list comprehension
    squares = [x**2 for x in range(1, 6)]
    print(f"Squares of 1-5: {squares}")

    # List comprehension with condition
    even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
    print(f"Even squares of 1-10: {even_squares}")

    # Dictionary comprehension
    square_dict = {x: x**2 for x in range(1, 6)}
    print(f"Square dictionary: {square_dict}")

    # Set comprehension
    vowel_set = {char for char in "hello world" if char in "aeiou"}
    print(f"Vowels in 'hello world': {vowel_set}")
    print()


def demonstrate_functions_and_oop():
    """Demonstrate functions and object-oriented programming."""
    print("Functions:")
    print("-" * 30)

    # Basic function
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    print(greet("World"))
    print(greet("Alice", "Good morning"))
    print(greet(greeting="Hi", name="Bob"))

    # Function with multiple return values
    def analyze_numbers(numbers):
        if not numbers:
            return 0, 0, 0
        total = sum(numbers)
        count = len(numbers)
        average = total / count
        return total, count, average

    sample_numbers = [10, 20, 30, 40, 50]
    total, count, avg = analyze_numbers(sample_numbers)
    print(f"Numbers: {sample_numbers}")
    print(f"Total: {total}, Count: {count}, Average: {avg:.2f}")

    # Lambda functions
    square = lambda x: x**2
    add = lambda x, y: x + y

    print(f"Square of 5: {square(5)}")
    print(f"Sum of 3 and 7: {add(3, 7)}")
    print()

    print("Object-Oriented Programming:")
    print("-" * 30)

    # Simple class
    class DataPoint:
        def __init__(self, x, y, label=""):
            self.x = x
            self.y = y
            self.label = label

        def distance_to(self, other):
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

        def __str__(self):
            return f"DataPoint({self.x}, {self.y}, '{self.label}')"

        def __repr__(self):
            return self.__str__()

    # Create instances
    point1 = DataPoint(0, 0, "origin")
    point2 = DataPoint(3, 4, "target")

    print(f"Point 1: {point1}")
    print(f"Point 2: {point2}")
    print(f"Distance between points: {point1.distance_to(point2):.2f}")

    # Inheritance
    class LabeledDataPoint(DataPoint):
        def __init__(self, x, y, label, category):
            super().__init__(x, y, label)
            self.category = category

        def get_info(self):
            return f"{self.label} ({self.category}) at ({self.x}, {self.y})"

    labeled_point = LabeledDataPoint(1, 2, "sample", "training")
    print(f"Labeled point: {labeled_point.get_info()}")
    print()


def demonstrate_file_io_and_error_handling():
    """Demonstrate file I/O and error handling."""
    print("File I/O Operations:")
    print("-" * 30)

    # Writing to a file
    filename = "sample_data.txt"
    try:
        with open(filename, "w") as file:
            file.write("Name,Age,City\n")
            file.write("Alice,30,Boston\n")
            file.write("Bob,25,New York\n")
            file.write("Charlie,35,Chicago\n")
        print(f"✅ Data written to {filename}")
    except IOError as e:
        print(f"❌ Error writing file: {e}")

    # Reading from a file
    try:
        with open(filename, "r") as file:
            content = file.read()
            print(f"File contents:\n{content}")
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
    except IOError as e:
        print(f"❌ Error reading file: {e}")

    # Reading line by line
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            print(f"Number of lines: {len(lines)}")
            for i, line in enumerate(lines):
                print(f"Line {i+1}: {line.strip()}")
    except IOError as e:
        print(f"❌ Error reading file: {e}")

    print()

    print("Error Handling:")
    print("-" * 30)

    # Try-except with different exception types
    def safe_divide(a, b):
        try:
            result = a / b
            return result
        except ZeroDivisionError:
            return "Error: Division by zero"
        except TypeError:
            return "Error: Invalid input types"
        except Exception as e:
            return f"Unexpected error: {e}"

    print(f"10 / 2 = {safe_divide(10, 2)}")
    print(f"10 / 0 = {safe_divide(10, 0)}")
    print(f"10 / 'a' = {safe_divide(10, 'a')}")

    # Context managers
    print("\nContext manager example:")
    try:
        with open(filename, "r") as file:
            first_line = file.readline().strip()
            print(f"First line: {first_line}")
    except IOError as e:
        print(f"❌ Error: {e}")

    # Clean up
    import os

    if os.path.exists(filename):
        os.remove(filename)
        print(f"✅ Cleaned up {filename}")
    print()


def demonstrate_data_science_packages():
    """Demonstrate essential Python packages for data science."""
    print("Essential Data Science Packages:")
    print("-" * 30)

    # Check available packages
    packages_to_check = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "jupyter",
    ]

    print("Package availability check:")
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Not installed")

    print()

    # Demonstrate numpy if available
    try:
        import numpy as np

        print("NumPy Demonstration:")
        print("-" * 20)

        # Create arrays
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([10, 20, 30, 40, 50])

        print(f"Array 1: {arr1}")
        print(f"Array 2: {arr2}")
        print(f"Sum: {arr1 + arr2}")
        print(f"Product: {arr1 * arr2}")
        print(f"Mean of array 1: {np.mean(arr1):.2f}")
        print(f"Standard deviation of array 1: {np.std(arr1):.2f}")

        # 2D array
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f"2D Matrix:\n{matrix}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix transpose:\n{matrix.T}")

    except ImportError:
        print("NumPy not available - install with: pip install numpy")

    print()

    # Demonstrate pandas if available
    try:
        import pandas as pd

        print("Pandas Demonstration:")
        print("-" * 20)

        # Create DataFrame
        data = {
            "Name": ["Alice", "Bob", "Charlie", "Diana"],
            "Age": [25, 30, 35, 28],
            "City": ["Boston", "NYC", "Chicago", "LA"],
            "Salary": [50000, 60000, 70000, 55000],
        }

        df = pd.DataFrame(data)
        print("Sample DataFrame:")
        print(df)
        print(f"\nDataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")

        # Basic operations
        print(f"\nAverage age: {df['Age'].mean():.1f}")
        print(f"Total salary: ${df['Salary'].sum():,}")
        print(f"City counts:\n{df['City'].value_counts()}")

    except ImportError:
        print("Pandas not available - install with: pip install pandas")

    print()

    print("Package Installation Commands:")
    print("-" * 30)
    print("pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter")
    print("pip install --upgrade pip  # Update pip first")
    print(
        "conda install numpy pandas matplotlib seaborn scikit-learn scipy jupyter  # If using conda"
    )


if __name__ == "__main__":
    main()
