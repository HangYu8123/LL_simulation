import numpy as np

import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv

def read_score( filename="score", obs_size = 4, action_size = 1):
    scores = []
    with open(filename+ ".csv", 'r') as file:
        reader = csv.reader(file,
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        cnt = 1
        for s in reader:
            #print(s)
            if cnt % 2 != 0:
                scores.append(float(s[0]))
            cnt+=1
            #print(cnt)
    return scores
steady = read_score("TAMER_STEADY")
steady_b = read_score("B")
steady_f = read_score("F")
steady_c = read_score("C")

fig, ax = plt.subplots()
data = [steady, steady_b, steady_f, steady_c]
ax.boxplot(data, notch= True)
ax.set_xticklabels(["None Noise", "Bias", "Focus", "Cons"])
plt.show()


print(steady)