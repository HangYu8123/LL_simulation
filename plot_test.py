import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()



def list_generator(mean, dis, number):
    return np.random.normal(mean, dis * dis, number)



girl20 = list_generator(1000, 29.2, 70)
boy20 = list_generator(800, 11.5, 80)
girl30 = list_generator(3000, 25.1056, 90)
boy30 = list_generator(1000, 19.0756, 100)

data = [girl20, boy20, girl30, boy30]
ax.boxplot(data, notch= True)
ax.set_xticklabels(["girl20", "boy20", "girl30", "boy30"])
plt.show()