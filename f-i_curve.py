import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# hodgkin huxley
gkmax = 0.36
vk = -77
gnamax = 1.2
vna = 50
gl = 0.003
vl = -54.387
cm = 0.010
niter = 10000
dt = 0.1
t = np.arange(1, niter + 1) * dt

def calculate_spiking_frequency(voltage, time):
    # Define the voltage threshold for spike detection
    threshold = -20  # in mV

    # Find the indices of the voltage samples that cross the threshold
    spike_indices = np.where(voltage > threshold)[0]

    # Calculate the inter-spike intervals
    spike_intervals = np.diff(time[spike_indices])

    # Calculate the mean inter-spike interval
    mean_isi = np.mean(spike_intervals)

    # Calculate the spiking frequency
    frequency = 1000 / mean_isi  # convert to Hz

    return frequency/10


input_levels = np.linspace(0.1,0.24,20)
frequencies = []
print(input_levels)
v = -64.9964
m = 0.0530
h = 0.5960
n = 0.3177
vhist = np.zeros(niter)
mhist = np.zeros(niter)
hhist = np.zeros(niter)
nhist = np.zeros(niter)


for i in input_levels:
    iapp = i * np.ones(niter)

    for j in range(niter):
        gna = gnamax * m**3 * h
        gk = gkmax * n**4
        gtot = gna + gk + gl
        vinf = (gna * vna + gk * vk + gl * vl + iapp[j]) / gtot
        tauv = cm / gtot
        v = vinf + (v - vinf) * np.exp(-dt / tauv)

        alpham = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
        betam = 4 * np.exp(-0.0556 * (v + 65))
        alphan = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
        betan = 0.125 * np.exp(-(v + 65) / 80)
        alphah = 0.07 * np.exp(-0.05 * (v + 65))
        betah = 1 / (1 + np.exp(-0.1 * (v + 35)))

        taum = 1 / (alpham + betam)
        tauh = 1 / (alphah + betah)
        taun = 1 / (alphan + betan)

        minf = alpham * taum
        hinf = alphah * tauh
        ninf = alphan * taun

        m = minf + (m - minf) * np.exp(-dt / taum)
        h = hinf + (h - hinf) * np.exp(-dt / tauh)
        n = ninf + (n - ninf) * np.exp(-dt / taun)

        vhist[j] = v
        mhist[j] = m
        hhist[j] = h
        nhist[j] = n
    frequency = calculate_spiking_frequency(vhist, t)
    frequencies.append(frequency)
    print(frequency)


# Plot the f-I curve
plt.figure()
plt.plot(input_levels, frequencies, "k|")
plt.xlabel("Input (mA)")
plt.ylabel("Frequency (Hz)")
plt.title("Average Frequency vs. Input")
plt.show()