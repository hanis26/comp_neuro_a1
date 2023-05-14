import matplotlib.pyplot as plt  # import matplotlib
import numpy as np  # import numpy
import time  # import time
import ipywidgets as widgets  # interactive display
from scipy.stats import pearsonr  # import pearson correlation
from scipy.signal import find_peaks

# hodgkin huxley


# at rest V=-64.9964 m=0.0530 h=0.5960 n=0.3177
# time msec, conductance mS/mm^2, voltage mV, capacitance uF/mm^2
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

iapp = 5 * np.ones(niter)

"""this is to simulate the post-inhibitory rebound
# Define the stimulation
iapp = np.zeros(niter)
iapp[0:500] = 0
iapp[500:1500] = -0.1
iapp[1500:] = 0"""


v = -64.9964
m = 0.0530
h = 0.5960
n = 0.3177

gnahist = np.zeros(niter)
gkhist = np.zeros(niter)
vhist = np.zeros(niter)
mhist = np.zeros(niter)
hhist = np.zeros(niter)
nhist = np.zeros(niter)

for i in range(niter):
    gna = gnamax * m**3 * h
    gk = gkmax * n**4
    gtot = gna + gk + gl
    vinf = (gna * vna + gk * vk + gl * vl + iapp[i]) / gtot
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

    vhist[i] = v
    mhist[i] = m
    hhist[i] = h
    nhist[i] = n


def calculate_spiking_frequency(voltage, time):
    # Determine a threshold value for detecting spikes
    threshold = -40  # in mV

    # Determine a minimum inter-spike interval for counting spikes
    min_spike_interval = 5  # in ms

    # Determine a maximum spike rate for calculating the window duration
    max_spike_rate = 200  # in Hz

    # Determine the time window for counting spikes
    spike_indices, _ = find_peaks(voltage, height=threshold)
    spike_intervals = np.diff(time[spike_indices])
    window_duration = max(
        min_spike_interval, 1000 / max_spike_rate, np.mean(spike_intervals)
    )

    # Count the number of voltage crossings of the threshold value
    spike_count = 0
    for i in range(len(spike_indices)):
        if time[spike_indices[i]] - time[spike_indices[0]] < window_duration:
            spike_count += 1
        else:
            break

    # Calculate the spiking frequency
    frequency = spike_count / window_duration * 1000  # convert to Hz
    return frequency


frequency = calculate_spiking_frequency(vhist, t)
print(f"Spiking frequency: {frequency:.2f} Hz")


# plot results
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# print("spikes:", vhist)
# print('time:',t)
axs[0].plot(t, vhist)
axs[0].set_title("voltage vs. time")

axs[1].plot(t, mhist, label="m")
axs[1].plot(t, hhist, label="h")
axs[1].plot(t, nhist, label="n")
axs[1].legend()

plt.show()
