import numpy as np

def gaussian_vector_field(
    tip_xyz=np.array([0.0, 0.0, 0.0]),
    tip_width=np.array([20.0, 20.0, 20.0]),
    grid_min=-25,
    grid_max=25,
    step=0.01,
    output_file="E_local.txt"
):

    x = np.arange(-7.5, 7.5 + step, step)
    y = np.arange(-5, 5 + step, step)
    z = np.arange(2.9, 3.1 + step, step)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    Dx = X - tip_xyz[0]
    Dy = Y - tip_xyz[1]
    Dz = Z - tip_xyz[2]

    norm = np.sqrt(Dx**2 + Dy**2 + Dz**2)

    prefactor = 4 * np.log(2)

    envelope = np.exp(
        -prefactor * (
            (Dx / tip_width[0])**2 +
            (Dy / tip_width[1])**2 +
            (Dz / tip_width[2])**2
        )
    )

    Ex = np.zeros_like(envelope)
    Ey = np.zeros_like(envelope)
    Ez = np.zeros_like(envelope)

    mask = norm > 1e-12

    Ex[mask] = envelope[mask] * Dx[mask] / norm[mask]
    Ey[mask] = envelope[mask] * Dy[mask] / norm[mask]
    Ez[mask] = envelope[mask] * Dz[mask] / norm[mask]

    out = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        Ex.ravel(),
        Ey.ravel(),
        Ez.ravel(),
    ])

    np.savetxt(
        output_file,
        out,
        fmt="%.6f %.6f %.6f %.10e %.10e %.10e"
    )

    print(f"Saved vector field to {output_file}")

if __name__ == "__main__": 
    gaussian_vector_field()

