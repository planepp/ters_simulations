#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Plot FHI-aims z dipole moment versus rel_shift_from_tip."
)
parser.add_argument(
    "directory",
    nargs="?",
    default=".",
    help="Root directory to search (default: current directory)",
)
args = parser.parse_args()

x = []
dipoles = []

for outfile in sorted(Path(args.directory).rglob("aims.out")):

    control = outfile.with_name("control.in")
    if not control.exists():
        print(f"Skipping {outfile}: no control.in")
        continue

    # Read x coordinate from rel_shift_from_tip
    xpos = None
    with open(control) as f:
        for line in f:
            if line.strip().startswith("rel_shift_from_tip"):
                fields = line.split()
                xpos = float(fields[1])
                break

    if xpos is None:
        print(f"Skipping {outfile}: rel_shift_from_tip not found")
        continue

    # Read z dipole from aims.out
    dipole = None
    with open(outfile) as f:
        for line in f:
            if "Total dipole moment in z-direction" in line:
                dipole = float(line.split()[-1])
                break

    if dipole is None:
        print(f"Skipping {outfile}: dipole not found")
        continue

    x.append(xpos)
    dipoles.append(dipole)

if not x:
    raise RuntimeError("No valid data found.")

# Sort by x position
pairs = sorted(zip(x, dipoles))
x, dipoles = zip(*pairs)

plt.figure(figsize=(7, 4))
plt.plot(x, dipoles, "o-", lw=1.5, label="Dipole")

# Vertical reference line
plt.axvline(x=26.4,color="red",linestyle="--",linewidth=1.5,label="Cell center",)
plt.axvline(x=0,color="red",linestyle="--",linewidth=1.5,label="Cell center",)
plt.axvline(x=13.2,color="k",linestyle="--",linewidth=1.5,label="Cell edge",)
plt.axvline(x=39.6,color="k",linestyle="--",linewidth=1.5,label="Cell edge",)

plt.xlabel("rel_shift_from_tip x (Å)")
plt.ylabel("Total dipole moment in z direction (eÅ)")
plt.title("Dipole vs. Tip Position")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
