import setuptools
import os

BASE_DIR = Path(__file__).parent
import setuptools
from pathlib import Path



with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open("requirements.txt") as f:
    requirements = [line.strip() for line in f]

pkginfo = {}
exec((BASE_DIR / "hpnevergrad" / "pkginfo.py").read_text(), None, pkginfo)

setuptools.setup(
    name="hnevergrad",
    version=pkginfo["__version__"],
    author="Xuehan Tan",
    author_email="tanxuehan@hotmail.com",
    description="A nevergrad extension for hpman",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/megvii-research/hpnevergrad",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)