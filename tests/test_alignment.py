import numpy as np

from xas_toolbox.io import XasMeasurement
from xas_toolbox.xas.alignment import align_scan, align_stack
from xas_toolbox.xas.data_generation import make_fake_spectrum, mix_scans


class TestAlignScans:
    def test_align_scan(self):
        as_data = make_fake_spectrum("C5H11AsO2", "As", "K", nscans=1)
        ref_data = make_fake_spectrum("As", "As", "K", nscans=1, exafs=False)
        as_data.refData.murefer = ref_data.mu

        as_aligned = align_scan(as_data)

        assert hasattr(as_aligned, "energy") and hasattr(as_aligned, "mu")
        assert hasattr(as_aligned.meta, "energy_shift")
        assert isinstance(as_aligned.meta.energy_shift, float)

    def test_align_stack(self):
        as_data_1 = make_fake_spectrum("AsHO2", "As", "K", nscans=1)
        as_data_2 = as_data_1
        as_data_2.energy += 100
        stacked_energy, stacked_mu = mix_scans(
            profiles=np.vstack((np.linspace(0, 1, 40), np.linspace(1, 0, 40))).T,
            absorption=[as_data_1.mu, as_data_2.mu],
            energy=[as_data_1.energy, as_data_2.energy],
        )

        def get_fn(valname: str):
            if valname == "energy":
                return stacked_energy
            if valname in ["mu", "mutrans"]:
                return stacked_mu

        stacked = XasMeasurement(get_value=get_fn, mode="transmission")

        aligned = align_stack(stacked)

        assert stacked.energy.shape[0] == len(as_data_1.energy)
        assert aligned.energy.ndim == 1
