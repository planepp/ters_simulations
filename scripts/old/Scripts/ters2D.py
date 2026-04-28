#!/usr/bin/env python
# coding: utf-8

# # Calculate and plot a TERS image

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

import ase.io
import sys
import os
sys.path.append(os.path.expanduser("~/.local/bin"))
import finite_field_ters as ffters

import ast
import argparse

plt.style.use('seaborn-v0_8-darkgrid')

def parse_modes(s):
    """Parse mode argument: single number, comma-separated list, or range x-y."""
    modes = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            modes.extend(range(start, end + 1))
        else:
            modes.append(int(part))
    return modes

parser = argparse.ArgumentParser(description="Calculate and plot a 2D TERS image")
parser.add_argument(
    "--mode",
    type=str,
    default=None,
    help="Mode index(es) to process: single int, comma-separated list, or range (e.g., '7-31' or '7,17,23')"
)
args = parser.parse_args()

# ## Read in calculation data
# After using the `ffters` module to set up the directory tree and (optionally) run the single points, it can be used to analyze the obtained data in a straightforward fashion as follows.

# %%
working_dir = Path(".")
geometry_file = next(working_dir.glob("calc2D*/tippos*/positive_displacement/field_on/geometry.in"))
system = ase.io.read(Path(geometry_file))
periodic = system.pbc.all()

### Tip-sample distance
working_dir = Path(".")
control_file = next(working_dir.glob("calc2D*/tippos*/positive_displacement/field_on/control.in"))
tip_height = None
with control_file.open() as f:
    for line in f:
        if line.strip().startswith("tip_molecule_distance"):
            tip_height = float(line.split()[-1])
            break  # stop after finding the first occurrence

print(f"Tip-sample distance = {tip_height} Å")

# %%
runters_file = Path("run-ters.py")
dq = 5e-3
efield = -1e-1
nbins = None
if runters_file.exists():
    with runters_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("dq"):
                dq = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("efield"):
                efield = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("bins"):
                value_str = line.split('=')[-1].strip()
                nbins = tuple(int(x) for x in ast.literal_eval(value_str))
else:
    print("run-ters.py not found in working directory. Assuming default dq, efield.")
    print("run-ters.py not found in working directory. Please add this file with the nbins = (n,m) tag.")
    exit(1)
    
print(f"dq = {dq}")
print(f"efield = {efield}")
print(f"nbins = {nbins}")

mode_idx = parse_modes(args.mode)
freqs = []
xyz_file = next(Path(".").glob("*ne.xyz"))  # pick the first match

with xyz_file.open() as f:
    for line in f:
        line = line.strip()
        if "stable frequency at" in line:
            # extract the frequency (second-to-last token)
            parts = line.split()
            freq = float(parts[3])  # parts[3] is the number before "1/cm"
            freqs.append(freq)
# print the frequency for the requested mode_idx
if mode_idx is None:
    # print all frequencies
    for i, fval in enumerate(freqs, 1):
        print(f"Mode {i}: {fval:.3f} 1/cm")
else:
    # mode_idx is a list of integers
    for idx in mode_idx:
        if 0 <= idx <= len(freqs):
            print(f"Mode {idx}: {freqs[idx]:.3f} 1/cm")
        else:
            print(f"Invalid mode_idx {idx}, file has {len(freqs)} modes")
ters = ffters.analyze_2d_ters(working_dir=Path('./'), mode_idx = mode_idx, efield=efield, dq=dq, nbins=nbins, periodic=periodic)


# scanning range of the TERS calculation
n = nbins[0]
m = nbins[1]
ex = np.array([-n, n, -m, m])

# ## Plot the TERS image
plt.imshow(ters['intensity'], origin='lower', extent=ex, cmap='viridis', interpolation='bilinear')
plt.colorbar(label=r'$(\mathrm{d}\alpha / \mathrm{d}Q)^2$ [e$^2$ $\mathrm{\AA}^2$ V$^{-2}$]')
plt.xlabel(r'$x$ [$\mathrm{\AA}$]')
plt.ylabel(r'$y$ [$\mathrm{\AA}$]')
if len(mode_idx) == 1: 
    plt.title(rf'TERS image d={tip_height} $\mathrm{{\AA}}$, f = {freqs[mode_idx[0]]} 1/cm')
else:
    plt.title(rf'TERS image d={tip_height} $\mathrm{{\AA}}$, {len(mode_idx)} modes included')
plt.tight_layout()
plt.savefig(f"2d_{mode_idx}.png", dpi=300)
plt.show();

