import numpy as np

a = numpy.array([0, 1, 2, 3])
print a

a = np.zeros(5)
print a

a += 1
print a

# Return evenly spaced values within a given interval
b = np.arange(0, 10, 2)
print b

print a + b
