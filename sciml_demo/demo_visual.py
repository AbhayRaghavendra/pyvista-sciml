import numpy as np
import pyvista as pv
import pyvista_sciml.sciml_accessor as sciml_accessor  # noqa: F401  (Registers .sciml on pv.DataSet)


def build_cylinder_wake_mesh() -> tuple[pv.UnstructuredGrid, dict[int, pv.PolyData]]:
    """Builds a 3D unstructured mesh around a cylinder with analytical vortex shedding."""
    # Create structured grid, carve out a central cylinder of radius 0.5, and convert to unstructured cells
    grid = pv.ImageData(dimensions=(90, 45, 5), spacing=(0.12, 0.12, 0.12), origin=(-2.5, -2.64, -0.24))
    cyl_dist = np.sqrt(grid.points[:, 0] ** 2 + grid.points[:, 1] ** 2)
    grid.point_data["r"] = cyl_dist
    mesh = grid.threshold(value=0.5, scalars="r")  # UnstructuredGrid with curved cylinder hole

    # Extract boundary patches: cylinder wall (r < 0.65) and left inlet (x == -2.5)
    outer_surf = mesh.extract_surface(algorithm="dataset_surface")
    wall_patch = outer_surf.threshold(value=[0.45, 0.68], scalars="r")
    inlet_patch = outer_surf.clip_box((-2.51, -2.49, -10, 10, -10, 10), invert=False)

    # Construct divergence-free streamfunction psi(x, y) -> u = d(psi)/dy, v = -d(psi)/dx
    # Guarantees exact physical mass conservation (div(u_true) == 0)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    r2 = np.maximum(x**2 + y**2, 0.25)
    wake_envelope = np.exp(-0.35 * y**2) * (1.0 / (1.0 + np.exp(-2.0 * (x - 0.5))))
    vortex_street = 0.45 * wake_envelope * np.sin(2.2 * x) * np.cos(1.8 * y)
    psi = y * (1.0 - 0.25 / r2) + vortex_street

    mesh.point_data["psi"] = psi
    grad_psi = mesh.compute_derivative(scalars="psi", gradient=True).point_data["gradient"]
    u_true = np.column_stack([grad_psi[:, 1], -grad_psi[:, 0], np.zeros_like(x)]).astype(np.float32)

    # Bernoulli-style pressure field
    speed2 = np.sum(u_true**2, axis=1)
    p_true = (0.5 * (1.0 - speed2)).astype(np.float32)

    # Simulate a typical AI surrogate prediction:
    # 1) Damps downstream vortex street (amplitude dissipation)
    # 2) Introduces localized divergence violation near x=1.2, y=0.2
    div_hotspot = 0.22 * np.exp(-((x - 1.2) ** 2 + (y - 0.2) ** 2) / 0.18)
    u_pred = u_true.copy()
    u_pred[:, 1] *= 0.65  # Damped vertical vortex oscillations
    u_pred[:, 0] += div_hotspot  # Non-divergence-free mass creation hotspot
    p_pred = (0.85 * p_true + 0.05 * div_hotspot).astype(np.float32)

    mesh.point_data["u_true"] = u_true
    mesh.point_data["u_pred"] = u_pred
    mesh.point_data["p_true"] = p_true
    mesh.point_data["p_pred"] = p_pred
    return mesh, {1: wall_patch, 2: inlet_patch}


def main() -> None:
    mesh, patches = build_cylinder_wake_mesh()

    # -------------------------------------------------------------------------
    # 1. PILLAR A: Extract graph topology & boundary masks
    # -------------------------------------------------------------------------
    graph = mesh.sciml.to_graph(patches=patches)
    mesh.point_data["node_type"] = graph["node_type"]
    print(f"[Pillar A] Nodes: {graph['coords'].shape[0]} | Deduplicated Directed Edges: {graph['edge_index'].shape[1]}")
    print(f"[Pillar A] Boundary counts -> Interior: {(graph['node_type']==0).sum()}, Wall: {(graph['node_type']==1).sum()}, Inlet: {(graph['node_type']==2).sum()}")

    # Visualize Pillar A (Graph edges + Boundary Node Types)
    pl_a = pv.Plotter(window_size=(1100, 500))
    pl_a.add_text("Pillar A: Extracted Graph Edges & Boundary Masks (0=Fluid, 1=Wall, 2=Inlet)", font_size=11)
    pl_a.add_mesh(mesh.slice(normal="z"), scalars="node_type", cmap=["#3b4cc0", "#e31a1c", "#33a02c"], show_edges=True)
    pl_a.view_xy()
    pl_a.show(screenshot="1_pillar_a_graph.png")

    # -------------------------------------------------------------------------
    # 2. PILLAR B: Audit physics conservation & integrated aerodynamic forces
    # -------------------------------------------------------------------------
    audited = mesh.sciml.audit_flow(pred_vel="u_pred", target_vel="u_true")

    wall_surf = audited.extract_surface().threshold(value=[0.45, 0.68], scalars="r")
    true_forces = wall_surf.sciml.integrate_drag_lift(pressure="p_true")
    pred_forces = wall_surf.sciml.integrate_drag_lift(pressure="p_pred")
    print(f"[Pillar B] Mean |Divergence| -> Truth: {np.mean(np.abs(audited['true_divergence'])):.4f} vs Pred: {np.mean(np.abs(audited['pred_divergence'])):.4f}")
    print(f"[Pillar B] Cylinder Drag (Cd) -> Truth: {true_forces['Cd']:.4f} vs Pred: {pred_forces['Cd']:.4f}")

    # -------------------------------------------------------------------------
    # 3. PILLAR C: Synchronized 3-panel comparisons (Vorticity & Mass Conservation)
    # -------------------------------------------------------------------------
    pl_vort = audited.sciml.compare(
        true_field="true_vorticity_mag",
        pred_field="pred_vorticity_mag",
        slice_normal="z",
        title_prefix="Vorticity:",
    )
    pl_vort.show(screenshot="2_pillar_c_vorticity.png")

    pl_div = audited.sciml.compare(
        true_field="true_divergence",
        pred_field="pred_divergence",
        slice_normal="z",
        title_prefix="Mass Divergence:",
    )
    pl_div.show(screenshot="3_pillar_c_divergence.png")


if __name__ == "__main__":
    main()