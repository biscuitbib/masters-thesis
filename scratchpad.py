import numpy as np

a = np.random.uniform(size=(2,2))

argmax = np.argmax(a, axis=0)

print(a, argmax, a[argmax])