import numpy as np
import matplotlib.pyplot as plt
from cclib.io import ccread

# ----------------------------
# Load FCHK file
# ----------------------------
fchk_file = "12.fchk"
data = ccread(fchk_file)

# ----------------------------
# Extract geometry
# ----------------------------
coords = data.atomcoords[-1]   # optimized geometry
numbers = data.atomnos         # atomic numbers

# ----------------------------
# Extract vibrations
# ----------------------------
freqs = data.vibfreqs
modes = data.vibdisps   # (n_modes, n_atoms, 3)

print("Atoms:", len(numbers))
print("Vibrational modes:", len(freqs))

# ----------------------------
# Choose mode to visualize
# ----------------------------
mode_id = 0
mode = modes[mode_id]

print(f"Plotting mode {mode_id} at {freqs[mode_id]:.2f} cm⁻¹")

# ----------------------------
# 2D projection (x-y plane)
# ----------------------------
x = coords[:, 0]
y = coords[:, 1]

dx = mode[:, 0]
dy = mode[:, 1]

# scale for visibility
scale = 2.0

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# atoms
ax.scatter(x, y, s=80, c="black", zorder=3)

# labels
for i in range(len(x)):
    ax.text(x[i], y[i], str(numbers[i]), fontsize=8)

# vibration arrows
for i in range(len(x)):
    ax.arrow(
        x[i], y[i],
        dx[i] * scale,
        dy[i] * scale,
        head_width=0.1,
        color="red",
        length_includes_head=True,
        zorder=2
    )

# styling
ax.set_title(f"Vibrational Mode {mode_id} @ {freqs[mode_id]:.1f} cm⁻¹")
ax.set_aspect("equal")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")

plt.tight_layout()
plt.show()
