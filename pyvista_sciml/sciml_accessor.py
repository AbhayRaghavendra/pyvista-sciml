from __future__ import annotations
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


@pv.register_dataset_accessor("sciml", pv.DataSet)
class SciMLAccessor:
    """PyVista accessor implementing SciML Graph ETL (A), Physics Auditing (B), and Visual Diagnostics (C)."""

    def __init__(self, pv_obj: pv.DataSet) -> None:
        self._obj = pv_obj

    # =========================================================================
    # PILLAR A: Mesh-to-Graph & Boundary Extraction
    # =========================================================================

    def boundary_mask(self, patches: dict[int, pv.DataSet], rtol: float = 1e-4) -> np.ndarray:
        """Tags mesh nodes by physical boundary condition (0=interior, 1=wall, 2=inlet, etc.)."""
        coords = np.ascontiguousarray(self._obj.points, dtype=np.float32)
        mask = np.zeros(len(coords), dtype=np.int32)
        if not patches:
            return mask

        diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        tol = max(diag * rtol, 1e-12)
        tree = cKDTree(coords)

        for label_id, patch_mesh in patches.items():
            dists, node_idx = tree.query(np.asarray(patch_mesh.points, dtype=np.float32), k=1)
            mask[node_idx[dists <= tol]] = label_id

        return mask

    def to_graph(self, patches: dict[int, pv.DataSet] | None = None) -> dict[str, np.ndarray]:
        """Extracts deduplicated bidirectional graph edges, coordinates, and boundary masks."""
        coords = np.ascontiguousarray(self._obj.points, dtype=np.float32)

        # Extract unique undirected cell edges and duplicate symmetrically for message passing
        edge_poly = self._obj.extract_all_edges()
        raw_lines = edge_poly.lines.reshape(-1, 3)[:, 1:3]
        undirected = np.unique(np.sort(raw_lines, axis=1), axis=0)

        edge_index = np.ascontiguousarray(
            np.vstack([
                np.concatenate([undirected[:, 0], undirected[:, 1]]),
                np.concatenate([undirected[:, 1], undirected[:, 0]]),
            ]),
            dtype=np.int32,
        )

        node_type = self.boundary_mask(patches) if patches else np.zeros(len(coords), dtype=np.int32)
        return {
            "coords": coords,         # (N, 3) float32, C-order
            "edge_index": edge_index, # (2, E) int32
            "node_type": node_type,   # (N,) int32
        }

    # =========================================================================
    # PILLAR B: Physics Conservation & Engineering Force Auditing
    # =========================================================================

    def audit_flow(self, pred_vel: str = "u_pred", target_vel: str = "u_true") -> pv.DataSet:
        """Computes velocity error, divergence (mass conservation), and vorticity magnitude."""
        out = self._obj.copy(deep=False)
        u_pred = np.asarray(out.point_data[pred_vel], dtype=np.float32)
        u_true = np.asarray(out.point_data[target_vel], dtype=np.float32)

        out.point_data["vel_abs_err"] = np.linalg.norm(u_pred - u_true, axis=-1)

        pred_deriv = out.compute_derivative(scalars=pred_vel, divergence=True, vorticity=True)
        true_deriv = out.compute_derivative(scalars=target_vel, divergence=True, vorticity=True)

        out.point_data["pred_divergence"] = pred_deriv.point_data["divergence"]
        out.point_data["true_divergence"] = true_deriv.point_data["divergence"]
        out.point_data["pred_vorticity_mag"] = np.linalg.norm(pred_deriv.point_data["vorticity"], axis=-1)
        out.point_data["true_vorticity_mag"] = np.linalg.norm(true_deriv.point_data["vorticity"], axis=-1)
        return out

    def integrate_drag_lift(
        self,
        pressure: str,
        rho: float = 1.0,
        u_inf: float = 1.0,
        ref_area: float = 1.0,
    ) -> dict[str, float]:
        """Integrates surface pressure into Drag (Cd) and Lift (Cl) coefficients."""
        # Ensure we have a PolyData surface even if the user passed a thresholded UnstructuredGrid
        surf = (
            self._obj
            if isinstance(self._obj, pv.PolyData)
            else self._obj.extract_surface(algorithm="dataset_surface")
        )
        surf = surf.point_data_to_cell_data(pass_point_data=True)
        surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
        surf = surf.compute_cell_sizes(length=False, area=True, volume=False)

        normals = np.asarray(surf.cell_data["Normals"], dtype=np.float64)
        areas = np.asarray(surf.cell_data["Area"], dtype=np.float64)[:, None]
        p_cell = np.asarray(surf.cell_data[pressure], dtype=np.float64)[:, None]

        force = -np.sum(p_cell * normals * areas, axis=0)
        q_inf = 0.5 * rho * (u_inf**2) * ref_area
        return {"Cd": float(force[0] / q_inf), "Cl": float(force[1] / q_inf)}
  

    # =========================================================================
    # PILLAR C: Synchronized 3-Panel Visual Diagnostics
    # =========================================================================

    def compare(
        self,
        true_field: str,
        pred_field: str,
        slice_normal: str | None = "z",
        title_prefix: str = "",
        off_screen: bool = False,
    ) -> pv.Plotter:
        """Opens a camera-linked 3-panel window (Truth vs. Prediction vs. Error) with locked color limits."""
        view = self._obj.slice(normal=slice_normal) if slice_normal else self._obj.copy(deep=False)
        err_key = f"_err_{pred_field}"
        view.point_data[err_key] = np.abs(
            np.asarray(view.point_data[pred_field]) - np.asarray(view.point_data[true_field])
        )

        # Lock color scale across Ground Truth & Prediction so damping is immediately visible
        clim = [float(np.min(view.point_data[true_field])), float(np.max(view.point_data[true_field]))]

        pl = pv.Plotter(shape=(1, 3), off_screen=off_screen, window_size=(1650, 520))
        panels = [
            (f"{title_prefix} Ground Truth", true_field, "viridis", clim),
            (f"{title_prefix} Surrogate Pred", pred_field, "viridis", clim),
            (f"{title_prefix} Absolute Error", err_key, "inferno", None),
        ]

        for col, (label, scalar, cmap, limits) in enumerate(panels):
            pl.subplot(0, col)
            pl.add_text(label, font_size=11)
            pl.add_mesh(view, scalars=scalar, cmap=cmap, clim=limits, show_edges=False)
            pl.view_xy()

        pl.link_views()
        return pl