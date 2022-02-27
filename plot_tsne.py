import matplotlib.pyplot as plt
import numpy as np

ppl = 100
(fig, ax) = plt.subplots(1, 1, figsize=(15, 15))

data_all = np.load(path)
for data in data_all:
    ax.scatter(data[0], data[1], color='red', s=50)
