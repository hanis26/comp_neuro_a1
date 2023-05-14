import numpy as np
import matplotlib.pyplot as plt

# Define the Hodgkin-Huxley model
C = 1.0  # membrane capacitance (uF/cm^2)
g_K = 36.0  # potassium conductance (mS/cm^2)
g_Na = 120.0  # sodium conductance (mS/cm^2)
g_L = 0.3  # leak conductance (mS/cm^2)
E_K = -77.0  # potassium reversal potential (mV)
E_Na = 50.0  # sodium reversal potential (mV)
E_L = -54.4  # leak reversal potential (mV)


def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))


def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65.0))


def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))


def beta_m(V):
    return 4.0 * np.exp(-0.0556 * (V + 65.0))


def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65.0))


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))


def compute_derivatives(V, n, m, h, I):
    dVdt = (I - g_K * n**4 * (V - E_K) - g_Na * m **
            3 * h * (V - E_Na) - g_L * (V - E_L)) / C
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    return dVdt, dndt, dmdt, dhdt


# Define simulation parameters
dt = 0.01  # time step (ms)
T = 1000.0  # total simulation time (ms)
t = np.arange(0.0, T, dt)  # time vector
V = np.zeros(len(t))  # membrane potential (mV)
n = np.zeros(len(t))  # potassium channel gating variable
m = np.zeros(len(t))  # sodium channel gating variable
h = np.zeros(len(t))  # sodium channel gating variable
I = 5.0  # input current (nA)

# Set initial conditions
V[0] = -65.0
n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
mu = 0.0  # mean of noise
sigma = 1.0  # standard deviation of noise
noise = np.random.normal(mu, sigma, len(t))  # generate Gaussian white noise

for i in range(1, len(t)):
    dVdt, dndt, dmdt, dhdt = compute_derivatives(
        V[i-1], n[i-1], m[i-1], h[i-1], I)
    V[i] = V[i-1] + dVdt * dt + noise[i]  # add noise to membrane potential
    n[i] = n[i-1] + dndt * dt
    m[i] = m[i-1] + dmdt * dt
    h[i] = h[i-1] + dhdt * dt
    if V[i] >= 30.0:  # spike detection
        V[i] = 30.0  # set membrane potential to spike threshold
        # update gating variables after spike
        n[i] = alpha_n(V[i]) / (alpha_n(V[i]) + beta_n(V[i]))
        m[i] = alpha_m(V[i]) / (alpha_m(V[i]) + beta_m(V[i]))
        h[i] = alpha_h(V[i]) / (alpha_h(V[i]) + beta_h(V[i]))


plt.plot(t, V, 'b')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Model with Gaussian White Noise')
plt.legend()
plt.show()
