import matplotlib.pyplot as plt
import numpy as np
from larch import Group
from sklearn.decomposition import PCA


def plot_snr_metrics(noise_poly, noise_k, kmax, group_0, group_sum) -> list:
    """
    Plot signal:noise metrics.
    """
    nscans = len(noise_poly)
    xplot = np.linspace(1, nscans + 1, nscans)  # noqa: E702

    figures = []

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_title(r"Estimated $\sigma_E$ (Polynomial smoothing)")
    ax.plot(xplot, noise_poly, "o", label="RMS Noise")
    ax.plot(
        xplot,
        noise_poly.max() / np.sqrt(xplot),
        label=r"$\frac{\sigma_{max}}{\sqrt{nscans}}$",
    )
    ax.legend()
    ax.set_ylabel(r"$\sigma_E$")
    ax.set_xlabel("Repetition Number")
    figures.append(fig)

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.plot(group_0.k, group_0.chi * group_0.k**2, label="Repetition 1")
    ax.plot(
        group_sum.k,
        group_sum.chi * group_sum.k**2,
        label=f"Summed {nscans} Repetitions",
    )
    ax.set_xlabel(r"k ($\AA^{-1}$)")
    ax.set_ylabel(r"$k^2\cdot\chi(k)$ $(\AA^{-2})$")
    ax.set_title(r"$k^2\cdot\chi(k)$")
    ax.legend()
    figures.append(fig)

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_title(r"Estimated $\epsilon_k$ (autobk)")
    ax.plot(xplot, noise_k, "o", label="Estimated noise (FFT)")
    ax.plot(
        xplot,
        noise_k.max() / np.sqrt(xplot),
        label=r"$\frac{\epsilon_{k_{max}}}{\sqrt{nscans}}$",
    )
    ax.legend()
    ax.set_ylabel(r"$\epsilon_k$")
    ax.set_xlabel("Repetition Number")
    figures.append(fig)

    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_title(r"Suggested $k_{max}$")
    ax.plot(xplot, kmax, "o", label=r"$k_{max}$")
    ax.set_ylabel(r"$k_{max}$ ($\AA^{-1}$)")
    ax.set_xlabel("Repetition Number")
    figures.append(fig)
    plt.show()

    print(f"Kmax sugggested {kmax[-1]}")
    return figures


def plot_repetition_metrics(data: Group) -> list:

    flat_mean = data.flat_avr
    flat_std = data.flat.std(axis=0)
    x, y, pre, post = data.energy, data.mu, data.pre_edge, data.post_edge

    figures = []

    fig, ax = plt.subplots(1, figsize=(10, 4))
    fig.suptitle(r"Average $\mu$")
    ax.plot(x, y)
    ax.set_xlabel("Energy (eV)")

    fig, ax = plt.subplots(2, 2, figsize=(10, 4), layout="tight")
    fig.suptitle(r"Average $\mu$ autobk \\n Edge Value = " + str(data.e0_avr))
    ax[0, 0].set_title("Pre and post-edge fit")
    ax[0, 0].plot(x, y)
    ax[0, 0].plot(x, pre, label="pre-edge")
    ax[0, 0].plot(x, post, label="post-edge")
    ax[0, 0].legend(fontsize="x-small")

    bkg, k, chi = data.bkg, data.k, data.chi

    ax[0, 1].set_title("Flattened average")
    ax[0, 1].plot(x, flat_mean)
    ax[0, 1].set_title("Flattened average")
    ax[0, 1].plot(x, flat_mean)
    ax[1, 0].plot(k, chi * k**2, label=r"$\chi$")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel(r"k ($\AA^{-1}$)")
    ax[1, 0].set_ylabel(r"$k^2\cdot\chi$ ($\AA^{-2}$)")
    ax[1, 1].plot(x, y, label="mu")
    ax[1, 1].plot(x, bkg, label="autobk")
    ax[1, 1].legend(fontsize="x-small")
    [a.set_xlabel("Energy (eV)") for a in [ax[0, 0], ax[0, 1], ax[1, 1]]]

    figures.append(fig)

    e0s = data.e0
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), layout="tight")
    ax[0].set_title("Edge Position")
    ax[0].set_xlabel("Repetition Number")
    ax[0].set_ylabel("Edge Position (eV)")
    ax[0].plot(np.arange(1, len(e0s) + 1, 1), e0s, "o")

    ax[1].set_title("Flattened Variation")
    ax[1].set_xlabel("Energy (eV)")
    ax[1].plot(x, flat_mean, label="average")
    ax[1].fill_between(
        x, flat_mean + flat_std, flat_mean - flat_std, color="r", label=r"$\pm\sigma^2$"
    )
    ax[1].legend()

    figures.append(fig)

    ncomp = 2
    model = PCA(n_components=ncomp)
    W = model.fit_transform(data.flat)  # noqa: N806
    H = model.components_  # noqa: E702, N806
    fig, ax = plt.subplots(ncomp, 2, figsize=(10, 8), layout="tight")
    fig.suptitle("PCA")
    for i in range(ncomp):
        ax[i, 0].set_title(f"PC{i + 1} Scores")
        ax[i, 0].set_xlabel("Repetition Number")
        ax[i, 0].scatter(np.arange(1, W.shape[0] + 1), W[:, i])

        ax[i, 1].set_title(f"PC{i + 1} Loadings")
        ax[i, 1].set_xlabel("Energy (eV)")
        ax[i, 1].plot(x, H[i, :])
    figures.append(fig)
    plt.show()
    return figures
