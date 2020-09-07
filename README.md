# easySE

The point is to be a CLI for making self-energies with sisl. 
This should be easier to do than to write the "same" boiler-plate code over and over.
It is MPI-parallelized.

If you take the scripts and make changes, please contribute the changes here!
That way we avoid having a hundred incompatible scripts lying around.

easySE for now only does RealspaceSI and not yet the RealspaceSE. Doing that should be trivial though. Contributions welcome. The same goes for spin polarization and using energy contours instead of linspace (relevant for TSGFs).

## Installation
easySE requires `sisl`, `mpi4py` and `zfpy`.
These are installed automatically if not already present.
Load your cluster's module versions of these before installing easySE to ensure that you get the 'right' versions (applies to mpi4py in particular).
Then run:
`pip3 install git+https://github.com/jonaslb/easySE.git`

## Usage
easySE for now only does RealspaceSI and not yet the RealspaceSE.
Use example: 

`easySE rssi -bulk-hs elec/RUN.fdf -surface-hs systs/RUN.fdf -surface-tile 5 5 1 -bulk-rsi-direction -C -mp-grid 33 33 -energy-linspace -1.5 1.5 100 -out-dir rssi-out -surf-nsc 3 3 1 -bulk-nsc 3 3 3 -matrix-fmt zfp -zfp-tolerance 1e-10`

Puts the self-energy data into a directory `rssi-out`. It is recommended that you use a separate directory for the output because many files are created.

To run in parallel with MPI, you should use the following syntax: ``mpirun python3 -m mpi4py -m easySE [args...]``. This ensures a proper abort across all processes in case an error occurs.

The script puts the self-energies into a directory specified with `-out-dir`. To convert the results to a TBTGF-file, use `easySE gfdir2gf output_dir`. Unfortunately this last step cannot be run in parallel (and it is usually IO limited, not CPU limited).
The GF-files are not compressed, so the size may increase dramatically.

By default, easySE stores the self-energies in a series of zfp-compressed files. This is a *lossy* compression although with the tolerance set by `-zfp-tolerance`. Even with a low tolerance such as `1e-10`, the size is often reduced by 80 %. The default is `1e-12` and it saves around 2/3 space.
If you require lossless compression, choose `npz` instead. Note that these files easily get very big.
You can do a test-run for your system with a single energy point and a 1x1 MP-grid to see how big the files get.
