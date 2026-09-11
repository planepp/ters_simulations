#!/usr/bin/env python3
"""
Plot out-of-plane vibrational modes as 2D xy maps, optionally labeled
with their point-group symmetry with respect to the Fe-centered
symmetry of the molecule.

Atoms are plotted at their xy positions and colored by the out-of-plane
(z) component of their displacement vector. Ag atoms are ignored
completely -- both for plotting and for the symmetry analysis -- since
they belong to the metal substrate and are not part of the symmetric
Fe complex.

Multiple modes can be plotted together in one figure with:
    - one vertical column of subplots
    - one shared colorbar
    - one shared color scale across all selected modes

Symmetry labeling (optional, --symmetry)
-----------------------------------------
When --symmetry is given, every requested mode is classified against
the D4h/C2v symmetry of the (Ag-free) molecule, centered on the Fe
atom:

  - For a pair of near-degenerate modes (frequencies closer than
    --degen-tol), C4 (90 deg about z through Fe) is tested FIRST,
    since that is the operation that actually distinguishes a true
    D4h Eg pair from an accidentally-close C2v B1/B2 pair (C2 and the
    mirrors alone cannot tell the two apart -- both have C2 ~ -1).
    Eg signature: trace(C4) ~ 0, C2 ~ -1 for both modes.

  - Otherwise the mode is classified alone under plain C2v
    (A1/A2/B1/B2) using C2 and the two auto-detected mirror planes
    (auto-detected from the 4 N atoms directly coordinating Fe).

Each plotted subplot is then titled:

    Mode <idx>, <freq> cm-1, <symmetry label>

instead of just the frequency.

Usage
-----
List available modes:
    python plot_vibration.py modes.xyz

List available modes with symmetry labels:
    python plot_vibration.py modes.xyz --symmetry

Plot one mode:
    python plot_vibration.py modes.xyz 12

Plot several modes, with symmetry labels in the titles:
    python plot_vibration.py modes.xyz 12 15 18 --symmetry

Save as PNG:
    python plot_vibration.py modes.xyz 12 15 18 --savepng

Use a fixed color scale:
    python plot_vibration.py modes.xyz 12 15 18 --zscale -0.3 0.3

Change atom size:
    python plot_vibration.py modes.xyz 12 15 --atoms-size 3000


Input XYZ format
----------------
The file is assumed to contain multiple XYZ frames.

Each frame has a header containing "frequency", for example:
    123.45 frequency ...

Each atom line is assumed to contain:
    symbol  x  y  z  dx  dy  dz

For example:
    C   -2.2782  -0.0563  -0.0113  -0.0160   0.0339  -0.0019

The first three numbers after the element are the atomic coordinates,
and the last three are the displacement vector.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ============================================================================
# Reading multi-mode XYZ files
# ============================================================================

def read_blocks(path):
    """
    Split a multi-frame XYZ file into (header, block_lines) tuples.
    """
    lines = Path(path).read_text().splitlines()

    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.isdigit():
            natoms = int(line)

            if i + 1 + natoms < len(lines):
                header = lines[i + 1]
                block = lines[i + 2:i + 2 + natoms]
                blocks.append((header, block))

            i += 2 + natoms

        else:
            i += 1

    return blocks


def parse_atom_line(line):
    """
    Parse:

        symbol x y z dx dy dz

    and return:

        symbol, position, displacement

    The symbol can be either a chemical symbol such as C, H, Ag,
    or an atomic number such as 6, 1, 47.
    """

    parts = line.split()

    raw_symbol = parts[0]

    # Convert atomic number to chemical symbol if necessary
    if raw_symbol.isdigit():
        symbol = chemical_symbols[int(raw_symbol)]
    else:
        symbol = raw_symbol

    x, y, z, dx, dy, dz = (
        float(v) for v in parts[1:7]
    )

    position = np.array([x, y, z])
    displacement = np.array([dx, dy, dz])

    return symbol, position, displacement


def get_modes(xyz_file):
    """
    Read all vibrational modes from the XYZ file.

    Returns a list of dictionaries containing:
        freq
        symbols
        positions
        displacements

    Ag atoms are dropped right here, so every downstream consumer
    (plotting AND symmetry analysis) automatically ignores them.
    """

    modes = []

    for header, block in read_blocks(xyz_file):

        # Only consider frames whose header contains "frequency"
        if "frequency" not in header.lower():
            continue

        # Extract numbers from the header
        floats = re.findall(
            r"[-+]?\d*\.\d+|\d+",
            header
        )

        if not floats:
            continue

        freq = float(floats[0])

        symbols = []
        positions = []
        displacements = []

        for line in block:

            sym, pos, disp = parse_atom_line(line)

            # Ignore Ag atoms completely (substrate, not part of the
            # symmetric Fe complex)
            if sym == "Ag":
                continue

            symbols.append(sym)
            positions.append(pos)
            displacements.append(disp)

        modes.append({
            "freq": freq,
            "symbols": np.array(symbols),
            "positions": np.array(positions),
            "displacements": np.array(displacements),
        })

    return modes


# ============================================================================
# Symmetry analysis (D4h Eg / C2v A1,A2,B1,B2), Ag atoms already excluded
# ============================================================================

def c4_matrix():
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def c2_matrix():
    return np.diag([-1, -1, 1])


def mirror_matrix(u):
    ux, uy = u
    M = np.eye(3)
    M[:2, :2] = [[2 * ux * ux - 1, 2 * ux * uy], [2 * ux * uy, 2 * uy * uy - 1]]
    return M


def build_mapping(pos, symbols, fe, matrix):
    transformed = (pos - fe) @ matrix.T + fe
    mapping = np.zeros(len(pos), dtype=int)
    for s in set(symbols):
        idx = np.where(symbols == s)[0]
        cost = cdist(transformed[idx], pos[idx])
        row, col = linear_sum_assignment(cost)
        mapping[idx[row]] = idx[col]
    return mapping


def apply_op(disp, matrix, mapping):
    transformed = disp @ matrix.T
    new_disp = np.zeros_like(disp)
    new_disp[mapping] = transformed
    return new_disp


def overlap(a, b):
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_mirror_axes(symbols, pos, fe):
    """Auto-detect the 2 mirror-plane directions from the 4 N atoms
    directly coordinating Fe (closest by distance), pairing up the ones
    ~180 deg apart across Fe."""
    n_idx = np.where(symbols == "N")[0]
    d = np.linalg.norm(pos[n_idx, :2] - fe[:2], axis=1)
    coord_n = n_idx[np.argsort(d)[:4]]
    vecs = pos[coord_n, :2] - fe[:2]
    angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    used, pairs = set(), []
    for i in range(4):
        if i in used:
            continue
        for j in range(4):
            if j != i and j not in used and abs(abs((angles[i] - angles[j] + 180) % 360 - 180) - 180) < 5:
                pairs.append((i, j))
                used.update((i, j))
                break
    u1 = vecs[pairs[0][0]] / np.linalg.norm(vecs[pairs[0][0]])
    u2 = vecs[pairs[1][0]] / np.linalg.norm(vecs[pairs[1][0]])
    return u1, u2


def classify_c2v(c2, sv1, sv2, tol):
    if min(abs(c2), abs(sv1), abs(sv2)) < tol:
        return "ambiguous"
    signs = tuple("+" if v > 0 else "-" for v in (c2, sv1, sv2))
    table = {("+", "+", "+"): "A1", ("+", "-", "-"): "A2",
             ("-", "+", "-"): "B1", ("-", "-", "+"): "B2"}
    return table.get(signs, "?")


def classify_modes(modes, degen_tol=1.0, eg_trace_tol=0.3, clean_tol=0.9):
    """
    Classify every mode in `modes` (Ag already stripped out) against
    the Fe-centered D4h/C2v symmetry of the molecule.

    Returns a list of labels, one per mode, in the same order as
    `modes` (so labels[i] corresponds to modes[i]).
    """

    symbols, pos = modes[0]["symbols"], modes[0]["positions"]

    fe_idx = np.where(symbols == "Fe")[0]
    if len(fe_idx) == 0:
        raise ValueError(
            "No Fe atom found (after removing Ag atoms) -- "
            "cannot determine the symmetry center."
        )
    fe = pos[fe_idx[0]]

    u1, u2 = find_mirror_axes(symbols, pos, fe)
    ops = {"C4": c4_matrix(), "C2": c2_matrix(), "sv1": mirror_matrix(u1), "sv2": mirror_matrix(u2)}
    mappings = {name: build_mapping(pos, symbols, fe, M) for name, M in ops.items()}

    def self_overlap(disp, opname):
        return overlap(apply_op(disp, ops[opname], mappings[opname]), disp)

    n = len(modes)
    labels = [None] * n

    i = 0
    while i < n:
        freq_i = modes[i]["freq"]

        # candidate Eg pair: test C4 between mode i and i+1 first
        if i + 1 < n and modes[i + 1]["freq"] - freq_i < degen_tol:
            di, dj = modes[i]["displacements"], modes[i + 1]["displacements"]
            t_di = apply_op(di, ops["C4"], mappings["C4"])
            t_dj = apply_op(dj, ops["C4"], mappings["C4"])
            trace_c4 = overlap(t_di, di) + overlap(t_dj, dj)
            c2_i, c2_j = self_overlap(di, "C2"), self_overlap(dj, "C2")

            if abs(trace_c4) < eg_trace_tol and c2_i < -0.9 and c2_j < -0.9:
                labels[i] = "Eg"
                labels[i + 1] = "Eg"
                i += 2
                continue

        # fall back: classify mode i alone under C2v
        di = modes[i]["displacements"]
        c2, sv1, sv2 = self_overlap(di, "C2"), self_overlap(di, "sv1"), self_overlap(di, "sv2")
        labels[i] = classify_c2v(c2, sv1, sv2, clean_tol)
        i += 1

    return labels


# ============================================================================
# Shared color normalization
# ============================================================================

def make_shared_norm(modes, zscale=None):
    """
    Create one color normalization shared by all selected modes.

    If zscale is supplied:
        --zscale VMIN VMAX

    then those values are used directly.

    Otherwise the color scale is symmetric around zero and is determined
    from the largest absolute dz among all atoms in all selected modes.
    (Ag atoms have already been removed at read time.)
    """

    # User-specified scale
    if zscale is not None:
        return Normalize(
            vmin=zscale[0],
            vmax=zscale[1]
        )

    max_abs = 0.0

    for mode in modes:

        dz = mode["displacements"][:, 2]

        if len(dz) > 0:
            max_abs = max(
                max_abs,
                np.abs(dz).max()
            )

    # Avoid an invalid Normalize if all dz happen to be zero
    if max_abs == 0:
        max_abs = 1.0

    return Normalize(
        vmin=-max_abs,
        vmax=max_abs
    )


# ============================================================================
# Plot one mode
# ============================================================================

def plot_vibration_mode(
    mode,
    norm,
    fig,
    ax,
    idx=None,
    label=None,
    xlim=None,
    ylim=None,
    atoms_size=2000,
    fontsize=18,
    cmap_name="RdBu",
):
    """
    Plot one vibrational mode into an existing matplotlib axis.

    Ag atoms have already been removed at read time.

    If `idx` is given, the title includes "Mode <idx>, ".
    If `label` is given (symmetry label), it is appended to the title.
    """

    positions = mode["positions"]
    dz = mode["displacements"][:, 2]

    # ------------------------------------------------------------------------
    # x/y limits
    # ------------------------------------------------------------------------

    if xlim is not None:
        idx_x = (
            (positions[:, 0] > xlim[0]) &
            (positions[:, 0] < xlim[1])
        )
    else:
        idx_x = np.ones(
            len(positions),
            dtype=bool
        )

    if ylim is not None:
        idx_y = (
            (positions[:, 1] > ylim[0]) &
            (positions[:, 1] < ylim[1])
        )
    else:
        idx_y = np.ones(
            len(positions),
            dtype=bool
        )

    # Final selection: inside x limits AND inside y limits
    selection = idx_x & idx_y

    # ------------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------------

    cmap = plt.get_cmap(cmap_name)

    colors = cmap(
        norm(dz[selection])
    )

    # Atomic numbers for marker sizes
    numbers = np.array([
        atomic_numbers[symbol]
        for symbol in mode["symbols"]
    ])

    # ------------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------------

    ax.scatter(
        positions[selection, 0],
        positions[selection, 1],
        color=colors,
        s=atoms_size * covalent_radii[numbers[selection]],
        edgecolors="k",
        linewidths=0.5,
    )

    # ------------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------------

    ax.set_aspect("equal")

    ax.set_xlabel(
        r"x ($\AA$)",
        fontsize=fontsize
    )

    ax.set_ylabel(
        r"y ($\AA$)",
        fontsize=fontsize
    )

    ax.tick_params(
        labelsize=fontsize
    )

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(5)
    )

    ax.yaxis.set_major_locator(
        plt.MaxNLocator(5)
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    # ------------------------------------------------------------------------
    # Title: "Mode <idx>, <freq> cm-1[, <label>]"
    # ------------------------------------------------------------------------

    title_parts = []
    if idx is not None:
        title_parts.append(f"Mode {idx}")
    title_parts.append(f"{mode['freq']:.1f} cm$^{{-1}}$")
    if label is not None:
        # Bold just the symmetry label via mathtext. Mathtext chokes on
        # spaces, so swap them for the math "\ " escape (only matters
        # for labels like "ambiguous" -- fine as-is, but future-proof).
        bold_label = label.replace(" ", r"\ ")
        title_parts.append(rf"$\mathbf{{{bold_label}}}$")

    ax.set_title(
        ", ".join(title_parts),
        fontsize=fontsize
    )

    # Apply limits if supplied
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


# ============================================================================
# Plot multiple modes
# ============================================================================

def plot_vibration_modes(
    modes,
    indices,
    labels=None,
    zscale=None,
    atoms_size=2000,
    fontsize=18,
    cmap_name="RdBu",
    xlim=None,
    ylim=None,
):
    """
    Plot several vibrational modes in one vertical figure.

    Layout:
        mode 1
        mode 2
        mode 3
        mode 4
        ...

    All modes share one colorbar and one color normalization.

    `labels`, if given, is a dict {mode_index: symmetry_label} used to
    annotate each subplot's title.
    """

    selected_modes = [
        modes[i]
        for i in indices
    ]

    n_modes = len(selected_modes)

    # ------------------------------------------------------------------------
    # Shared color normalization
    # ------------------------------------------------------------------------

    norm = make_shared_norm(
        selected_modes,
        zscale=zscale
    )

    # ------------------------------------------------------------------------
    # One vertical column
    # ------------------------------------------------------------------------

    ncols = n_modes
    nrows = 1

    # Height is scaled with number of modes.
    #
    # This is intentionally a tall figure because the requested layout
    # is always n x 1.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0, 5.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------------
    # Plot each mode
    # ------------------------------------------------------------------------

    for ax, idx, mode in zip(axes, indices, selected_modes):

        label = labels.get(idx) if labels is not None else None

        plot_vibration_mode(
            mode=mode,
            norm=norm,
            fig=fig,
            ax=ax,
            idx=idx,
            label=label,
            xlim=xlim,
            ylim=ylim,
            atoms_size=atoms_size,
            fontsize=fontsize,
            cmap_name=cmap_name,
        )

    # ------------------------------------------------------------------------
    # One shared colorbar
    # ------------------------------------------------------------------------

    cmap = plt.get_cmap(cmap_name)

    sm = cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        orientation="vertical",

        # Small colorbar
        fraction=0.012,
        pad=0.03,
        shrink=0.30,
        aspect=35,
    )

    cbar.ax.set_ylabel(
        r"$\delta z$ ($\AA$)",
        fontsize=fontsize
    )

    cbar.ax.tick_params(
        labelsize=fontsize
    )

    cbar.set_ticks(
        np.linspace(
            norm.vmin,
            norm.vmax,
            5
        )
    )

    cbar.ax.yaxis.set_major_formatter(
        "{x:.2f}"
    )

    return fig, axes


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot out-of-plane displacement maps for one or more "
            "vibrational modes, optionally labeled with their "
            "point-group symmetry."
        )
    )

    # ------------------------------------------------------------------------
    # Input XYZ file
    # ------------------------------------------------------------------------

    parser.add_argument(
        "xyz",
        help="Input multi-mode XYZ file"
    )

    # ------------------------------------------------------------------------
    # Mode indices
    #
    # nargs="*" allows:
    #
    #   modes.xyz
    #   modes.xyz 12
    #   modes.xyz 12 15 18
    # ------------------------------------------------------------------------

    parser.add_argument(
        "idx",
        type=int,
        nargs="*",
        help="One or more mode indices"
    )

    # ------------------------------------------------------------------------
    # Atom size
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--atoms-size",
        type=float,
        default=2000,
        help=(
            "Marker size scale factor "
            "(default: 2000)"
        )
    )

    # ------------------------------------------------------------------------
    # Color scale
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--zscale",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help=(
            "Fixed colorbar range in Angstrom, "
            "e.g. --zscale -0.3 0.3. "
            "If omitted, the scale is symmetric around zero "
            "and determined from the largest |dz| across all "
            "selected non-Ag atoms."
        )
    )

    # ------------------------------------------------------------------------
    # Symmetry labeling
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--symmetry",
        action="store_true",
        help=(
            "Classify each mode against the Fe-centered D4h/C2v "
            "symmetry of the molecule (Ag atoms ignored) and show the "
            "label (Eg, A1, A2, B1, B2, ...) in the mode list and in "
            "each subplot's title."
        )
    )

    parser.add_argument(
        "--degen-tol",
        type=float,
        default=1.0,
        help=(
            "cm^-1: modes closer than this are tested for an Eg pair "
            "(default: 1.0). Only used with --symmetry."
        )
    )

    parser.add_argument(
        "--eg-trace-tol",
        type=float,
        default=0.3,
        help=(
            "|trace(C4)| below this counts as Eg (default: 0.3). "
            "Only used with --symmetry."
        )
    )

    parser.add_argument(
        "--clean-tol",
        type=float,
        default=0.9,
        help=(
            "Overlap magnitude above this counts as a clean C2v "
            "eigenvector (default: 0.9). Only used with --symmetry."
        )
    )

    # ------------------------------------------------------------------------
    # Save PNG
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--savepng",
        action="store_true",
        help="Save the figure as a PNG"
    )

    # ------------------------------------------------------------------------
    # Suppress interactive display
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Do not open an interactive plot window. "
            "Useful for batch runs or headless machines "
            "(e.g. combined with --savepng)."
        )
    )

    args = parser.parse_args()

    # =========================================================================
    # Read modes (Ag atoms already dropped)
    # =========================================================================

    modes = get_modes(args.xyz)

    # =========================================================================
    # Symmetry classification (optional)
    # =========================================================================

    labels = None
    if args.symmetry:
        try:
            label_list = classify_modes(
                modes,
                degen_tol=args.degen_tol,
                eg_trace_tol=args.eg_trace_tol,
                clean_tol=args.clean_tol,
            )
        except ValueError as e:
            print(f"Symmetry analysis failed: {e}")
            sys.exit(1)
        labels = {i: label_list[i] for i in range(len(modes))}

    # =========================================================================
    # List available modes
    # =========================================================================

    print(
        f"Found {len(modes)} vibrational modes "
        f"in {args.xyz}:"
    )

    for i, mode in enumerate(modes):

        if labels is not None:
            print(
                f"  {i:3d}: "
                f"{mode['freq']:.2f} cm-1  "
                f"{labels[i]}"
            )
        else:
            print(
                f"  {i:3d}: "
                f"{mode['freq']:.2f} cm-1"
            )

    # =========================================================================
    # No mode specified -> only list modes
    # =========================================================================

    if not args.idx:

        print(
            "\nNo mode index specified — "
            "exiting after listing available modes."
        )

        sys.exit(0)

    # =========================================================================
    # Validate mode indices
    # =========================================================================

    for idx in args.idx:

        if not (0 <= idx < len(modes)):

            print(
                f"Invalid mode index {idx}. "
                f"Must be between 0 and {len(modes) - 1}."
            )

            sys.exit(1)

    # =========================================================================
    # Remove duplicate indices while preserving order
    # =========================================================================

    selected_indices = list(
        dict.fromkeys(args.idx)
    )

    # =========================================================================
    # Print selected modes
    # =========================================================================

    print("\nSelected modes:")

    for idx in selected_indices:

        if labels is not None:
            print(
                f"  {idx:3d}: "
                f"{modes[idx]['freq']:.3f} cm-1  "
                f"{labels[idx]}"
            )
        else:
            print(
                f"  {idx:3d}: "
                f"{modes[idx]['freq']:.3f} cm-1"
            )

    # =========================================================================
    # Create figure
    # =========================================================================

    fig, axes = plot_vibration_modes(
        modes=modes,
        indices=selected_indices,
        labels=labels,
        zscale=args.zscale,
        atoms_size=args.atoms_size,
    )

    # =========================================================================
    # Save
    # =========================================================================

    if args.savepng:

        mode_string = "-".join(
            str(i)
            for i in selected_indices
        )

        out_png = (
            f"{Path(args.xyz).stem}"
            f"_mode{mode_string}.png"
        )

        # Do NOT use bbox_inches="tight" here.
        #
        # Keeping the original figure bounding box preserves the
        # carefully constructed n x 1 layout when saving.
        fig.savefig(
            out_png,
            dpi=300,
            pad_inches=0.2,
        )

        print(
            f"\nSaved {out_png}"
        )

    # =========================================================================
    # Display
    # =========================================================================

    if not args.no_show:
        plt.show()
