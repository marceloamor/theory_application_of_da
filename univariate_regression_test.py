import numpy as np
import matplotlib.pyplot as plt
test1 =np.random.normal(0,1,1000000)
test2 =np.random.normal(0,1,100)

print(test1.mean())
print(test2.mean())

residuals = test1 - test1.mean()

print(residuals)

# distribution of residuals
plt.hist(residuals)
# plot normal distribution line on same graph

plt.plot(np.linspace(-5,5,1000000), np.exp(-np.linspace(-5,5,1000000)**2/2)/np.sqrt(2*np.pi), color='red')
plt.savefig('residuals.png')

