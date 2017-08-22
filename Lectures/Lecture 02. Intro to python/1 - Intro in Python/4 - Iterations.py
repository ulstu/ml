import numpy


def Summ(a, b):
    return a + b


x = numpy.array([0, 1, 2, 3])
y = numpy.array([1, 2, 3, 4])

# Transpose as matrix
arrays = numpy.array([x, y]).T
print arrays
exit()

z = []

for i, j in arrays:
    res = Summ(i, j)
    z.append(res)

print z
exit()

if 0 not in z:
    print 'zero not in ', z
