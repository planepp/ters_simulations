import shutil
from pathlib import Path
import numpy as np

def read_grid_coords(mode_dir: Path):
    """
    Read tip positions from all tippos_*/positi/fieldon/controlin/ folders.
    Returns array of shape (n_points, 2) with (x, y) coordinates,
    and the corresponding tippos index for each point.
    """
    coords = []
    indices = []

    for tippos_dir in sorted(mode_dir.glob('tippos_*')):
        control_file = tippos_dir / 'positive_displacement' / 'field_on' / 'control.in'
        if not control_file.exists():
            continue

        with open(control_file) as f:
            for line in f:
                if line.strip().startswith('rel_shift_from_tip'):
                    parts = line.split()
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
                    indices.append(int(tippos_dir.name.split('_')[1]))
                    break

    return np.array(coords), np.array(indices)


def add_tippos(mode_dir: Path, new_idx: int, x: float, y: float, template_idx: int = 0):
    """
    Create a new tippos folder by copying a template and updating the position.

    mode_dir:     path to ters2d/mode_xxx
    new_idx:      index for the new tippos folder
    x, y:         new position
    template_idx: tippos to copy from (default: 0)
    """
    src = mode_dir / f'tippos_{template_idx:03d}'
    dst = mode_dir / f'tippos_{new_idx:03d}'

    if dst.exists():
        raise FileExistsError(f"{dst} already exists.")

    shutil.copytree(src, dst)

    # update controlin files (there may be more than one)
    for control_file in dst.rglob('control.in'):
        lines = control_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith('rel_shift_from_tip'):
                parts = line.split()
                new_lines.append(f"{parts[0]}      {x:.6f} {y:.6f}")
            else:
                new_lines.append(line)
        control_file.write_text('\n'.join(new_lines) + '\n')

    print(f"Created {dst} at position ({x}, {y})")

mode_dir = Path('ters2d/mode_063')
coords, indices = read_grid_coords(mode_dir)
# get unique sorted positions per axis


# overlay the tip positions
i = np.where(indices == 52)[0][0]
print(coords[i])

add_tippos(mode_dir, new_idx=68, x=0.75, y=4.125)
add_tippos(mode_dir, new_idx=69, x=0.75, y=7.875)
add_tippos(mode_dir, new_idx=70, x=0.75, y=9.75)
add_tippos(mode_dir, new_idx=71, x=0.75, y=12)
