import numpy as np
import pyvista as pv
from tqdm import tqdm


def main():
    directories = ("high_fluid", "medium_fluid", "low_fluid")
    volume_meshes = []
    surface_meshes = []
    print("Extracting surface normals from volume meshes...")
    for directory in tqdm(directories):
        mesh = pv.read(f"{directory}/geometry.vtu")
        volume_meshes.append(mesh)
        surf = mesh.extract_surface(algorithm=None)
        surface_meshes.append(surf)
        surf = surf.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
        )

        orig_ids = surf.point_data["vtkOriginalPointIds"]  # Get the original point IDs from the surface mesh
        normals = np.zeros((mesh.n_points, 3))  # Create an array to hold the normals for the volume mesh
        normals[orig_ids] = surf.point_data["Normals"]  # Assign the normals to

        mesh.point_data["Normals"] = normals  # Add the normals to the volume mesh

    # Save meshes as D files and as vtp for checking
    print("Saving volume meshes with normals...")
    for i in tqdm(range(len(volume_meshes))):
        volume_meshes[i].save(f"{directories[i]}/geometry.vtu")
        np.savetxt(
            f"{directories[i]}/meshes/normals.D",
            volume_meshes[i].point_data["Normals"],
            header=f"{volume_meshes[i].n_points} 3",
            comments="",
            fmt="%.16f",
        )


if __name__ == "__main__":
    main()
