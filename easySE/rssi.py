#!/usr/bin/env python3
"""This script wraps sisl RealSpaceSI, runs it in mpi-parallel over energies and saves the result to a directory.
Use `gfdir2gf outdir myelec.tbtgf` to convert the parallel results into the tbtgf needed for Siesta/tbtrans.
If you invoke it directly, use `mpirun [-n X] python3 -m mpi4py make_rssi.py [args]`.
You can supply the `-example-submit`, then an example_submit.lsf file is dumped to your working dir for hpc systems.
It is best if the number of energies is divisible by the number of processors. Otherwise, one processor will do much less work.
""" 
import sisl as si
import numpy as np
from mpi4py import MPI
import argparse
from pathlib import Path
from datetime import datetime
from math import ceil
import sys
from contextlib import contextmanager
import gc
try:
    import zfpy
except ImportError:
    zfpy = None


comm = MPI.COMM_WORLD
rank = comm.rank

global _last_rank
_last_rank = 0
@contextmanager
def pick_a_rank(barrier=False):
    global _last_rank
    this_rank = _last_rank
    _last_rank += 1
    _last_rank %= comm.size
    yield this_rank
    if barrier:
        comm.Barrier()


def mprint(*args, printer=0, skip=False, **kwargs):
    if skip or rank != printer:
        return
    print(f"{datetime.now():%H:%M:%S}:", *args, **kwargs)


def file_path(path):
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"File {p} does not exist!")
    return p


def existing_path(path):
    p = Path(path)
    p.mkdir(exist_ok=True, parents=True)
    return p


class SubmitFileAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        kwargs["nargs"] = 0
        super().__init__(*args, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        del sys.argv[sys.argv.index("-example-submit")]
        opts = " ".join(sys.argv[1:])
        submitstr = (f"""\
#!/usr/bin/env bash
#BSUB -J make_rssi
#BSUB -q hpc
#BSUB -W 24:0
#BSUB -R "select[avx2] span[block=20]"
#BSUB -n 60
#BSUB -M 6GB
#BSUB -B
#BSUB -N
ulimit -s unlimited
echo SLOTS : $LSB_DJOB_NUMPROC
echo OMP NUM THREADS : $OMP_NUM_THREADS
echo LIST OF HOSTS:
echo ${{LSB_MCPU_HOSTS}}
echo The current stack limit is \"$( ulimit -s )\".
echo ------------------------------------------------------

mpirun python3 -m mpi4py {__file__} {opts}
"""
        )
        p = Path() / "example_submit.lsf"
        p.write_text(submitstr)
        print(f"Wrote example submit file to {p}. Exiting without doing any work. Bye!")
        sys.exit(0)


def add_args(parser):
    a = parser.add_argument
    a("-bulk-hs", type=file_path, required=True, help="The fully bulk hamiltonian.")
    a("-surface-hs", type=file_path, required=True, help="The surface hamiltonian.")
    a("-surface-tile", type=int, nargs=3, default=(1, 1, 1), help="Tiling/unfolding - give 3 integers")
    a("-bulk-rsi-direction", default="-C", help="The semi-infinite direction. Default '-C'")
    a("-mp-grid", required=True, type=int, nargs="+", help=
        "Monkhorst-Pack grid number of points. Same number of args as k-axes. Will be reduced slightly thanks to TRS")
    a("-energy-linspace", required=True, nargs=3, type=float, help="The energy grid as a linspace, 3 args: lower upper npoints")
    a("-energy-imag", default=1e-3, type=float, help="Imaginary energy (default 1e-3)")
    a("-out-dir", type=existing_path, required=True, help="Directory to save into")
    a("-k-axes", nargs="+", default=[0, 1], type=int, help="Axes over which to k-sample. Default (0,1)")
    a("-surf-nsc", nargs=3, type=int, default=None, help="Give 3 ints. nsc for surf hs is set to this.")
    a("-bulk-nsc", nargs=3, type=int, default=None, help="Give 3 ints. nsc for bulk hs is set to this.")
    matrix_fmts = ["npz"]
    if zfpy is not None:
        matrix_fmts += ["zfp"]
    a("-matrix-fmt", choices=matrix_fmts, default=matrix_fmts[-1],
        help="Use either the numpy npz format or the lossy-compressed zfp. Default uses zfp if zfpy is available.")
    a("-zfp-tolerance", type=float, default=1e-12,
        help="zfp lossy compression tolerance for when matrix-fmt=zfp. Default=1e-12. Saves around 2/3 disk space.")
    a("-example-submit", action=SubmitFileAction, help="Output example submit script and stop")
    return parser


def get_argparser():
    parser = argparse.ArgumentParser()
    return add_args(parser)


def run(
    bulk_hs,
    surface_hs,
    surface_tile,
    bulk_rsi_direction,
    mp_grid,
    energy_linspace,
    energy_imag,
    out_dir,
    k_axes,
    surf_nsc,
    bulk_nsc,
    matrix_fmt,
    zfp_tolerance,
    example_submit
):
    mprint("Reading hamiltonians")
    He = si.get_sile(bulk_hs).read_hamiltonian()
    Hs = si.get_sile(surface_hs).read_hamiltonian()
    
    if bulk_nsc is not None:
        He.set_nsc(bulk_nsc)
    if surf_nsc is not None:
        Hs.set_nsc(surf_nsc)

    with pick_a_rank() as r:
        mprint(f"Writing hamiltonians ({r})", printer=r)
        if rank == r:
            He.write(out_dir / "bulk_hamiltonian.nc")
            Hs.write(out_dir / "surface_hamiltonian.nc")

    rsi = si.physics.RecursiveSI(He, bulk_rsi_direction)
    rssi = si.physics.RealSpaceSI(rsi, Hs, k_axes, unfold=surface_tile)

    coupling_geom, se_indices = rssi.real_space_coupling(True)
    with pick_a_rank() as r:
        mprint(f"Saving coupling geometry ({r})", printer=r)
        if rank == r:
            coupling_geom.write(out_dir / "coupling_geometry.nc")
            np.save(out_dir / "coupling_geometry_indices", se_indices)

    parenths = rssi.real_space_parent()
    new_order = np.concatenate((se_indices, np.delete(np.arange(parenths.na), se_indices)))
    parenths = parenths.sub(new_order)
    with pick_a_rank() as r:
        mprint(f"Saving parent geometry ({r})", printer=r)
        if rank == r:
            parenths.geometry.write(out_dir / f"full_geometry.fdf")
            parenths.write(out_dir / f"full_geometry.nc")
    del parenths

    mp = np.array([1, 1, 1], dtype=int)
    mp[k_axes] = mp_grid
    rssi.set_options(bz=si.MonkhorstPack(coupling_geom, mp))

    E, dE = np.linspace(*energy_linspace[:2], int(energy_linspace[2]), retstep=True)
    nE = len(E)
    E = E + 1j*energy_imag
    with pick_a_rank() as r:
        mprint(f"Saving energy grid ({r})", printer=r)
        if rank == r:
            np.save(out_dir / "energy_grid", E)
            si.io.TableSile(out_dir / "energy_grid.table", "w").write_data(E.real, E.imag, np.full(E.shape, dE))

    nperrank = ceil(nE / comm.size)
    local_eidx = np.arange(rank * nperrank, min((rank + 1) * nperrank, nE))
    mprint("Energy grid distribution:", nperrank, "energy points per processor.")
    if (nE % nperrank):
        mprint(f"One process only has {nE % nperrank} energy points.")

    mprint(f"To assess processing progress, use `echo $(( 100 * $(find {out_dir} -type f -name 'SE_E*.npz' | wc -l) / {nE} ))%`")
    mprint(f"Note that these files due to the parallelism are probably created in bunches of {comm.size} and each bunch may take long to finish.")
    for ie, e in zip(local_eidx, E[local_eidx]):
        se = rssi.self_energy(e, bulk=True, coupling=True)
        mprint(f"SE{ie:>03d} calculated", printer=rank)
        if matrix_fmt == "npz":
            np.savez_compressed(out_dir / f"SE_E{ie:>03d}.npz", se)
        elif matrix_fmt == "zfp":
            bs = zfpy.compress_numpy(se.real, tolerance=zfp_tolerance)
            (out_dir / f"SE_E{ie:>03d}_REAL.zfp").write_bytes(bs)
            bs = zfpy.compress_numpy(se.imag, tolerance=zfp_tolerance)
            (out_dir / f"SE_E{ie:>03d}_IMAG.zfp").write_bytes(bs)
            del bs
        del se
        mprint(f"SE{ie:>03d} saved", printer=rank)
        gc.collect()
    mprint(f"MPI-rank {rank} done.", printer=rank)
    comm.Barrier()
    mprint((
        f"All done! Use `gfdir2gf {out_dir} myelec.tbtgf` to convert the parallel results"
        " into the tbtgf needed for Siesta/tbtrans."
    ))


if __name__ == "__main__":
    args = get_argparser().parse_args()
    mprint(f"Parsed args: {args}")
    run(**args)
    
    


