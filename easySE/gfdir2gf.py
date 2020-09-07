import sisl as si
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from math import ceil
import sys
from functools import partial
try:
    import zfpy
except ImportError:
    zfpy = None
from tqdm import tqdm


class ForceFormatAction(argparse.Action):
    def __init__(self, **kwargs):
        self.fmtval = kwargs.get("option_strings")[0][2:].split("-")[1]
        kwargs["nargs"] = 0
        super().__init__(**kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, self.fmtval)


class InputDirAction(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, values)
        setattr(ns, "output", ns.inputdir / "BULKSE.TBTGF")


def add_args(parser):
    a = parser.add_argument
    a("inputdir", type=Path, action=InputDirAction)
    a("--output", "-o", type=Path, help="Default is inputdir/BULKSE.TBTGF")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--format", default="auto", choices=("auto", "zfp", "npz"))
    g.add_argument("--force-zfp", action=ForceFormatAction, dest="format")
    g.add_argument("--force-npz", action=ForceFormatAction, dest="format")
    return parser


def get_parser():
    parser = argparse.ArgumentParser()
    return add_args(parser)


def get_se_npz(ie, dir=None):
    return np.load(dir / f"SE_E{ie:>03d}.npz")["arr_0"]


def get_se_zfp(ie, dir=None, loadinto=None):
    loadinto.real[:, :] = zfpy.decompress_numpy((dir / f"SE_E{ie:>03d}_REAL.zfp").open("rb").read())
    loadinto.imag[:, :] = zfpy.decompress_numpy((dir / f"SE_E{ie:>03d}_IMAG.zfp").open("rb").read())
    return loadinto


def get_loader(fmt, dir, norb):
    if fmt == "npz":
        return partial(get_se_npz, dir=dir)
    elif fmt == "zfp":
        reusable_array = np.empty((norb, norb), dtype=np.complex128)
        return partial(get_se_zfp, dir=dir, loadinto=reusable_array)
    raise NotImplementedError(f"Unknown format {fmt}")


def run(
    inputdir,
    format,
    output,
):
    HV = si.get_sile(inputdir / "coupling_geometry.nc").read_hamiltonian()
    E = si.io.TableSile(inputdir / "energy_grid.table").read_data()[:2].T.copy().view(dtype=np.complex128)
    
    if format == "auto":
        if any(".zfp" in p.name for p in inputdir.iterdir()):
            format = "zfp"
        elif any(".npz" in p.name for p in inputdir.iterdir()):
            format = "npz"
        else:
            raise ValueError("Couldn't figure out format of input directory. Are there any self energies there?")
    get_se = get_loader(format, inputdir, HV.no)

    with si.io.tbtgfSileTBtrans(output, "w") as out:
        out.write_header(si.BrillouinZone(HV), E)
        for ispin, is_k, k, e in tqdm(out):
            if is_k:
                opts = {'dtype': np.complex128, 'format': 'array'}
                out.write_hamiltonian(HV.Hk(k=k, **opts), HV.Sk(k=k, **opts))
            ie = np.argmin(np.abs(E-e))
            out.write_self_energy(get_se(ie))
    print("Done!")

if __name__ == "__main__":
    args = get_parser().parse_args()
    run(**args)
