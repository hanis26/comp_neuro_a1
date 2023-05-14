import matplotlib.pyplot as plt  # import matplotlib
import numpy as np  # import numpy
import time  # import time
import ipywidgets as widgets  # interactive display
from scipy.stats import pearsonr  # import pearson correlation

fig_w, fig_h = (6, 4)
plt.rcParams.update({"figure.figsize": (fig_w, fig_h)})


def default_pars(**kwargs):
    pars = {}

    ### typical neuron parameters###

    pars["V_th"] = -55.0  # spike threshold [mV]
    pars["V_reset"] = -75.0  # reset potential [mV]
    pars["tau_m"] = 10.0  # membrane time constant [ms]
    pars["g_L"] = 10.0  # leak conductance [nS]
    pars["V_init"] = -65.0  # initial potential [mV]
    pars["V_L"] = -75.0  # leak reversal potential [mV]
    pars["tref"] = 2.0  # refractory time (ms)

    ### simulation parameters ###
    pars["T"] = 400.0  # Total duration of simulation [ms]
    pars["dt"] = 0.1  # Simulation time step [ms]

    ### external parameters if any ###
    for k in kwargs:
        pars[k] = kwargs[k]

    pars["range_t"] = np.arange(
        0, pars["T"], pars["dt"]
    )  # Vector of discretized time points [ms]

    return pars


def run_LIF(pars, I):
    """
    Simulate the LIF dynamics with external input current

    Expects:
    pars       : parameter dictionary
    I          : input current [pA]. The injected current here can be a value or an array

    Returns:
    rec_spikes : spike times
    rec_v      : membrane potential
    """

    # Set parameters
    V_th, V_reset = pars["V_th"], pars["V_reset"]
    tau_m, g_L = pars["tau_m"], pars["g_L"]
    V_init, V_L = pars["V_init"], pars["V_L"]
    dt, range_t = pars["dt"], pars["range_t"]
    Lt = range_t.size
    tref = pars["tref"]
    # Initialize voltage and current
    v = np.zeros(Lt)
    v[0] = V_init
    I = I * np.ones(Lt)
    tr = 0.0
    # simulate the LIF dynamics
    rec_spikes = []  # record spike times
    for it in range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:  # reset voltage and record spike event
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt
        # calculate the increment of the membrane potential
        dv = (-(v[it] - V_L) + I[it] / g_L) * (dt / tau_m)

        # update the membrane potential
        v[it + 1] = v[it] + dv

    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes


"""In the following we will inject direct current and white noise to study the response of an LIF neuron.

Constant current

following cell to run the LIF neuron when receiving a DC current, and see the voltage response of the LIF neuron."""

pars = default_pars()
v, rec_spikes = run_LIF(pars, I=500.)
plt.plot(pars["range_t"], v, "b")
plt.xlim(0, 100)
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.title("Spikes")
plt.show()

def find_spiking_frequency(pars, I):
    """
    Calculate the spiking frequency of an LIF neuron given its parameters and an injected current.

    Expects:
    pars       : parameter dictionary
    I          : input current [pA]. The injected current here can be a value or an array

    Returns:
    freq       : spiking frequency [Hz]
    """

    v, rec_spikes = run_LIF(pars, I)
    if len(rec_spikes) <= 1:
        freq = 0
    else:
        freq = 1 / np.mean(np.diff(rec_spikes))

    return freq

pars = default_pars()
I = 500.  # pA
freq = find_spiking_frequency(pars, I)
print(f"Spiking frequency: {freq:.2f} Hz")


"""Gaussian white noise (GWN) current"""


def my_GWN(pars, sig, myseed=False):
    """
    Expects:
    pars       : parameter dictionary
    sig        : noise amplitute
    myseed     : random seed. int or boolean

    Returns:
    I          : Gaussian white noise input
    """

    # Retrieve simulation parameters
    dt, range_t = pars["dt"], pars["range_t"]
    Lt = range_t.size

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # generate GWN
    I = sig * np.random.randn(Lt) / np.sqrt(dt / 1000.0)

    return I


"""generate the GWN current and plot it out."""

pars = default_pars()
sig_ou = 0.5

I_GWN = my_GWN(pars, sig=sig_ou, myseed=1998)
plt.plot(pars["range_t"], I_GWN, "b")
plt.xlabel("Time (ms)")
plt.ylabel(r"$I_{GWN}$ (pA)")
plt.title("Gaussian White Noise")
plt.show()

v, rec_spikes = run_LIF(pars, I=I_GWN + 250.0)
plt.plot(pars["range_t"], v, "b")
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.xlim(100, 200)
plt.title('Noisy Spikes')
plt.show()
