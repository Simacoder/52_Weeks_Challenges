# Python code
import ctypes

# Load the shared library
lib = ctypes.CDLL('./add_floats.so')

# Define the argument and return types for the function
lib.add_floats.argtypes = [ctypes.c_float, ctypes.c_float]
lib.add_floats.restype = ctypes.c_float

# Convert python float to c_float type 
a = ctypes.c_float(3.5)
b = ctypes.c_float(2.2)

# Call the C function
result = lib.add_floats(a, b)
print(result)
# 5.7