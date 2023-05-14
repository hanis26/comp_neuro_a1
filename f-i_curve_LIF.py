import matplotlib.pyplot as plt  # import matplotlib
import numpy as np  # import numpy
import time  # import time
import ipywidgets as widgets  # interactive display
from scipy.stats import pearsonr  # import pearson correlation
import LIFmodel

# Define a range of input currents
I_range = np.arange(800, 1400, 100)

# Calculate the spiking frequency for each input current
freqs = []
for I in I_range:
    freq = LIFmodel.find_spiking_frequency(LIFmodel.pars, I)
    freqs.append(freq)

# Plot the f-I curve
plt.plot(I_range, freqs, 'b.')
plt.xlabel('Input current (pA)')
plt.ylabel('Spiking frequency (Hz)')
plt.title("Average Frequency vs. Input")
plt.show()