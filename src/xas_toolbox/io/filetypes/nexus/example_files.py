"""
Quick fixes for using example data from outdated nexus files.
"""

from .nxs_reader import NxsReader


class LegacyB18Reader(NxsReader):
    """
    Quick fix for reading in some legacy data from a summer school.

    Attributes:
        path (Path): Path to file.
        elementMeta (dict): Paths to element metadata (currently <b>symbol,
                                                        edge, reference,
                                                        ref_edge</b>).
        transParams (dict): Paths to transmission data: <b>mutrans, energy</b>.
        fluorParams (dict): `None`
        referParams (dict): Paths to reference data: <b>murefer</b>.
        mcaParams (dict): `None`
    """  # noqa: E501

    def __init__(self, path):
        super().__init__(path)

        self.monoParams["energy"] = "/entry1/qexafs_counterTimer01/qexafs_energy"

        self.transParams["mutrans"] = "/entry1/qexafs_counterTimer01/lnI0It"

        self.referParams["murefer"] = "/entry1/qexafs_counterTimer01/lnItIref"

        self.elementMeta["symbol"] = ["entry1/before_scan/QEXAFS_Parameters", "element"]
        self.elementMeta["edge"] = ["entry1/before_scan/QEXAFS_Parameters", "edge"]

    def _get_ScanData(self, val):  # noqa: N802
        return super()._get_ScanData(val)
