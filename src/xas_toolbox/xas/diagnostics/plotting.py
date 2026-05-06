import matplotlib.pyplot as plt
import numpy as np

def plot_snr_metrics(noise_poly, noise_k, kmax,
                     group_0, group_sum):
    """
    Plot signal:noise metrics.
    """
    nscans = len(noise_poly); xplot = np.linspace(1, nscans+1, nscans)

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_title(r"Estimated $\sigma_E$ (Polynomial smoothing)")
    ax.plot(xplot, noise_poly, "o", label="RMS Noise")
    ax.plot(xplot, noise_poly.max()/np.sqrt(xplot), 
            label=r"$\frac{\sigma_{max}}{\sqrt{nscans}}$")
    ax.legend(); ax.set_ylabel(r"$\sigma_E$"); ax.set_xlabel("Repetition Number")
    
    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.plot(group_0.k, group_0.chi*group_0.k**2, label="Repetition 1")
    ax.plot(group_sum.k, group_sum.chi*group_sum.k**2, label=f"Summed {nscans} Repetitions")
    ax.set_xlabel(r"k ($\AA^{-1}$)"); ax.set_ylabel(r"$k^2\cdot\chi(k)$ $(\AA^{-2})$")
    ax.set_title(r'$k^2\cdot\chi(k)$'); ax.legend()

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_title(r"Estimated $\epsilon_k$ (autobk)")
    ax.plot(xplot, noise_k, "o", label="Estimated noise (FFT)")
    ax.plot(xplot, noise_k.max()/np.sqrt(xplot), label = r"$\frac{\epsilon_{k_{max}}}{\sqrt{nscans}}$")
    ax.legend(); ax.set_ylabel(r"$\epsilon_k$"); ax.set_xlabel("Repetition Number")

    fig, ax = plt.subplots(1, figsize=(10,4))
    ax.set_title(r"Suggested $k_{max}$")
    ax.plot(xplot, kmax, "o", label=r"$k_{max}$")
    ax.set_ylabel(r"$k_{max}$ ($\AA^{-1}$)"); ax.set_xlabel("Repetition Number")
    plt.show()

    print(f"Kmax sugggested {kmax[-1]}")
