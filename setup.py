#!/usr/bin/env python3
from pathlib import Path
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="easySE",
        version="0.0.1",
        description="Easy command-line interfaces for making self-energies",
        long_description=(Path(__file__).parent / "README.md").open().read(),
        long_description_content_type="text/markdown",
        url="https://github.com/jonaslb/easySE",
        author="Jonas L. Bertelsen",
        license="LGPLv3",
        packages=["easySE"],
        entry_points={"console_scripts": ["easySE = easySE.cli:main"]},
        install_requires=["sisl", "zfpy", "mpi4py"],
        zip_safe=False,
    )
