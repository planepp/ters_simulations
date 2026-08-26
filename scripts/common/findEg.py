#!/usr/bin/env python3
"""
Identify genuine D4h Eg modes by testing the C4 (90 deg) rotation directly,
not just C2/mirrors. For every pair of near-degenerate modes, builds the
2x2 matrix representing how C4 acts within that 2-mode subspace and checks
it against the Eg character-table signature:
    trace(C4) ~ 0   (each mode fully rotates into its partner)
    C2 overlap ~ -1 for both modes
A real 2D irrep pair will show this regardless of which two axes were used
to build the mode vectors - it's a property of the pair, not of our mirror
choice.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

FNAME = "fepc.xyz"
DEGEN_TOL = 0.6   # cm^-1: modes within this of each other are candidate pairs


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


def normalized_overlap(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    blocks = read_blocks(FNAME)
    symbols, pos = blocks[0][1], blocks[0][2]
    fe = pos[symbols == "Fe"][0]

    mapping_c4 = build_mapping(pos, symbols, fe, c4_matrix())
    mapping_c2 = build_mapping(pos, symbols, fe, c2_matrix())

    freqs = [b[0] for b in blocks]

    print(f"{'pair':>10} {'freqs':>20} {'split':>8}  {'C4 trace':>9} {'C2_i':>6} {'C2_j':>6}  verdict")
    n = len(blocks)
    for gi in range(n - 1):
        gj = gi + 1
        if blocks[gj][0] - blocks[gi][0] >= DEGEN_TOL:
            continue
        di, dj = blocks[gi][3], blocks[gj][3]

        t_di = apply_op(di, c4_matrix(), mapping_c4)
        t_dj = apply_op(dj, c4_matrix(), mapping_c4)
        M00 = normalized_overlap(t_di, di)
        M11 = normalized_overlap(t_dj, dj)
        trace_c4 = M00 + M11

        c2_i = normalized_overlap(apply_op(di, c2_matrix(), mapping_c2), di)
        c2_j = normalized_overlap(apply_op(dj, c2_matrix(), mapping_c2), dj)

        is_eg = abs(trace_c4) < 0.3 and c2_i < -0.9 and c2_j < -0.9
        verdict = "Eg (D4h)" if is_eg else "near-degenerate, not Eg"
        print(f"{gi:>4}-{gj:<4} {blocks[gi][0]:8.3f}/{blocks[gj][0]:<8.3f} "
              f"{blocks[gj][0]-blocks[gi][0]:8.3f}  {trace_c4:9.2f} {c2_i:6.2f} {c2_j:6.2f}  {verdict}")


if __name__ == "__main__":
    main()
