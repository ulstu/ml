import random as r
import numpy as np
import matplotlib.pyplot as plt
import math
x = np.linspace(0, 5, 50)
y1 = [math.sin(i) + r.uniform(-0.2, 0.2) for i in x]
y2 = [math.sin(i) for i in x]
#print(x, y)

plt.subplot(2, 1, 1)
plt.plot(x, y1, 'ro')
plt.xlabel("x")
plt.ylabel("y1")
plt.subplot(2, 1, 2)
plt.plot(x, y2, 'b--')



