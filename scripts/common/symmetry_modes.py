#!/usr/bin/env python3
"""
Classify vibrational modes (blocks of "symbol x y z dx dy dz", one block
per mode, atom count + frequency comment as header) with respect to the
Fe-centered symmetry of the molecule.

For every pair of near-degenerate modes, C4 (90 deg about z through Fe) is
tested FIRST, since that's the operation that actually distinguishes a
true D4h Eg pair from an accidentally-close C2v B1/B2 pair -- C2 and the
mirrors alone cannot tell the two apart (both have C2 = -1). Only if the
C4 test rules out Eg does the mode fall back to being classified under
plain C2v (A1/A2/B1/B2) using C2 + the two auto-detected mirror planes.

Eg signature (D4h character table): trace(C4) ~ 0, C2 ~ -1 for both modes.
C2v signature: each mode's own C2/sv1/sv2 overlaps are individually ~ +-1.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- edit these for your case ---
#FNAME = "fepc.xyz"
#FNAME = "aligned_nosubstrate.xyz"
#FNAME = "ag111_nosubstrate.xyz"
FNAME = "rotated_nosubstrate.xyz"
FREQ_RANGE = (340, 400)    # only show modes in this window (cm^-1)
DEGEN_TOL = 0.1            # cm^-1: modes closer than this are tested for Eg
EG_TRACE_TOL = 0.3         # |trace(C4)| below this counts as Eg
CLEAN_TOL = 0.7            # overlap magnitude above this counts as a clean C2v eigenvector
# ----------------------------------


def read_blocks(fname):
    lines = open(fname).readlines()
    natoms = int(lines[0])
    blocksize = natoms + 2
    nblocks = len(lines) // blocksize
    blocks = []
    for b in range(nblocks):
        off = b * blocksize
        freq = float(lines[off + 1].split("frequency at")[1].split("1/cm")[0])
        symbols, pos, disp = [], [], []
        for l in lines[off + 2: off + 2 + natoms]:
            p = l.split()
            symbols.append(p[0])
            pos.append([float(x) for x in p[1:4]])
            disp.append([float(x) for x in p[4:7]])
        blocks.append((freq, np.array(symbols), np.array(pos), np.array(disp)))
    return blocks


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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def main():
    blocks = read_blocks(FNAME)
    symbols, pos = blocks[0][1], blocks[0][2]
    fe = pos[symbols == "Fe"][0]

    u1, u2 = find_mirror_axes(symbols, pos, fe)
    ops = {"C4": c4_matrix(), "C2": c2_matrix(), "sv1": mirror_matrix(u1), "sv2": mirror_matrix(u2)}
    mappings = {name: build_mapping(pos, symbols, fe, M) for name, M in ops.items()}

    def self_overlap(disp, opname):
        return overlap(apply_op(disp, ops[opname], mappings[opname]), disp)

    n = len(blocks)
    print(f"{'idx':>4} {'freq(1/cm)':>11}   {'label':<10} {'details'}")

    i = 0
    while i < n:
        freq_i = blocks[i][0]
        show_i = FREQ_RANGE[0] <= freq_i <= FREQ_RANGE[1]

        # candidate Eg pair: test C4 between mode i and i+1 first
        if i + 1 < n and blocks[i + 1][0] - freq_i < DEGEN_TOL:
            di, dj = blocks[i][3], blocks[i + 1][3]
            t_di = apply_op(di, ops["C4"], mappings["C4"])
            t_dj = apply_op(dj, ops["C4"], mappings["C4"])
            trace_c4 = overlap(t_di, di) + overlap(t_dj, dj)
            c2_i, c2_j = self_overlap(di, "C2"), self_overlap(dj, "C2")

            if abs(trace_c4) < EG_TRACE_TOL and c2_i < -0.9 and c2_j < -0.9:
                if show_i or (FREQ_RANGE[0] <= blocks[i + 1][0] <= FREQ_RANGE[1]):
                    print(f"{i:4d}/{i+1:<4d} {freq_i:11.3f}/{blocks[i+1][0]:<8.3f} "
                          f"{'Eg (D4h)':<10} trace(C4)={trace_c4:.2f}, C2={c2_i:.2f}/{c2_j:.2f}")
                i += 2
                continue

        # fall back: classify mode i alone under C2v
        di = blocks[i][3]
        c2, sv1, sv2 = self_overlap(di, "C2"), self_overlap(di, "sv1"), self_overlap(di, "sv2")
        label = classify_c2v(c2, sv1, sv2, CLEAN_TOL)
        if show_i:
            print(f"{i:4d}      {freq_i:11.3f}   {label:<10} C2={c2:.2f}, sv1={sv1:.2f}, sv2={sv2:.2f}")
        i += 1


if __name__ == "__main__":
    main()
