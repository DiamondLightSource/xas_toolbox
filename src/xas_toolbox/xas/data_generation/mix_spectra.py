import numpy as np

from xas_toolbox.utils.maths.filter import normalise_matrix, resize_with_replacement
from xas_toolbox.xas.alignment.align_stack import get_target_energy, interp_stack


def mix_scans(
    profiles: np.ndarray,
    absorption: list[np.ndarray],
    energy: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mix a series of absorption values with their own respective energies according
    to profiles provided.

    Arguments:
        profiles (np.ndarray): Concentration values for the components to follow.
        absorption (list[np.ndarray]): List of absorption values to mix.
        energy (list[np.ndarray] | np.ndarray): Single/list of energy values\
              corresponding to the absorption values.

    Returns:
        tuple (tuple): tuple containing:
            energy_out (np.ndarray): Common energy axis for all spectra.
            out (np.ndarray): ndarray of mixed spectra.
    """
    if profiles.shape[1] != len(absorption):
        profiles = profiles.T
        if profiles.shape[1] != len(absorption):
            raise ValueError(
                "Absorption profile shape doesn't match number of spectra."
            )

    profiles = normalise_matrix(profiles, axis=-1)

    # need to check if the lengths of absorption + energies match first:
    if len(list({len(ab) for ab in absorption})) > 1:
        if isinstance(energy, list):
            absorption = resize_with_replacement(absorption)
            energy = resize_with_replacement(energy)
        else:
            raise ValueError("Provide all energy arrays for spectra.")
    else:
        absorption = np.array(absorption)
        energy = np.array(energy)
    if energy.ndim > 1:
        energy_out = get_target_energy(np.array(energy))
        absorption = interp_stack(energy, absorption, energy_out)
    else:
        energy_out = energy

    # absorption = interp_stack(energy, absorption, energy_out)

    out = np.zeros((profiles.shape[0], absorption.shape[1]))

    # ...
    for i in range(profiles.shape[0]):
        conc = profiles[i, :]
        out[i, :] = np.sum(np.multiply(absorption.T, conc), axis=1)

    return energy_out, out
