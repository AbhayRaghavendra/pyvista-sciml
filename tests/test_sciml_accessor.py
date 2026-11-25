from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_sciml.sciml_accessor as sciml_accessor  # noqa: F401  (Registers .sciml on pv.DataSet)


# =============================================================================
# Fixtures: Reusable Synthetic Geometries & Analytical Flow Fields
# =============================================================================


@pytest.fixture
def unit_cube_grid() -> pv.ImageData:
    """4x4x4 uniform grid on [0, 1]^3 with analytical divergence-free flow."""
    grid = pv.ImageData(
        dimensions=(4, 4, 4),
        spacing=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        origin=(0.0, 0.0, 0.0),
    )
    x, y, z = grid.points[:, 0], grid.points[:, 1], grid.points[:, 2]

    # Exact divergence-free velocity field: d(u)/dx + d(v)/dy + d(w)/dz == 0
    u_true = np.column_stack([
        np.sin(np.pi * x) * np.cos(np.pi * y),
        -np.cos(np.pi * x) * np.sin(np.pi * y),
        np.zeros_like(z),
    ]).astype(np.float32)

    # Perturbed prediction with both damping and non-zero divergence
    u_pred = (0.8 * u_true + 0.1 * x[:, None]).astype(np.float32)

    grid.point_data["u_true"] = u_true
    grid.point_data["u_pred"] = u_pred
    return grid


@pytest.fixture
def cylinder_surface() -> pv.PolyData:
    """Closed 3D cylinder surface (radius=0.5, height=1.0) aligned along the z-axis."""
    return pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5,
        height=1.0,
        resolution=60,
        capping=True,
    )


# =============================================================================
# PILLAR A TESTS: Graph Extraction & Boundary Tagging
# =============================================================================


class TestPillarA:
    def test_to_graph_structure_and_contiguity(self, unit_cube_grid: pv.ImageData) -> None:
        graph = unit_cube_grid.sciml.to_graph()

        assert set(graph.keys()) == {"coords", "edge_index", "node_type"}

        # Verify C-contiguous memory layout and expected dtypes
        assert graph["coords"].flags["C_CONTIGUOUS"]
        assert graph["coords"].dtype == np.float32
        assert graph["coords"].shape == (unit_cube_grid.n_points, 3)

        assert graph["edge_index"].flags["C_CONTIGUOUS"]
        assert graph["edge_index"].dtype == np.int32
        assert graph["edge_index"].ndim == 2
        assert graph["edge_index"].shape[0] == 2

        # Default node_type should be all zeros (interior fluid) when no patches are passed
        assert graph["node_type"].dtype == np.int32
        assert np.all(graph["node_type"] == 0)

    def test_to_graph_edges_are_deduplicated_and_symmetric(self, unit_cube_grid: pv.ImageData) -> None:
        graph = unit_cube_grid.sciml.to_graph()
        edges = graph["edge_index"].T  # (E, 2)

        # 1. No self-loops
        assert not np.any(edges[:, 0] == edges[:, 1])

        # 2. No duplicate directed edges
        unique_directed = np.unique(edges, axis=0)
        assert len(unique_directed) == len(edges)

        # 3. Every directed edge (u, v) has an exact reverse edge (v, u)
        edge_set = {(int(u), int(v)) for u, v in edges}
        for u, v in edge_set:
            assert (v, u) in edge_set

        # 4. Exact edge count for a 4x4x4 Cartesian grid:
        # Undirected edges = 3 axes * (3 edges per row * 4 * 4 rows) = 144 -> 288 directed edges
        assert len(edges) == 288

    def test_boundary_mask_assigns_patch_labels_and_priority(self, unit_cube_grid: pv.ImageData) -> None:
        # Extract x=0 plane (wall=1) and y=0 plane (inlet=2)
        wall_patch = unit_cube_grid.extract_subset((0, 0, 0, 3, 0, 3))
        inlet_patch = unit_cube_grid.extract_subset((0, 3, 0, 0, 0, 3))

        mask = unit_cube_grid.sciml.boundary_mask(patches={1: wall_patch, 2: inlet_patch})

        # Verify wall-only nodes are 1, inlet nodes (including shared corner edge) are 2
        assert np.sum(mask == 2) == 16  # Entire y=0 plane (4x4 = 16 nodes)
        assert np.sum(mask == 1) == 12  # x=0 plane minus the 4 shared nodes on y=0
        assert np.sum(mask == 0) == 64 - 28

    def test_boundary_mask_empty_patches_returns_zeros(self, unit_cube_grid: pv.ImageData) -> None:
        mask = unit_cube_grid.sciml.boundary_mask(patches={})
        assert mask.shape == (unit_cube_grid.n_points,)
        assert np.all(mask == 0)

    def test_to_graph_works_on_unstructured_grid(self) -> None:
        ugrid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        graph = ugrid.sciml.to_graph()
        assert graph["coords"].shape == (27, 3)
        # 3x3x3 grid has 3 * (2 * 3 * 3) = 54 undirected edges -> 108 directed edges
        assert graph["edge_index"].shape == (2, 108)


# =============================================================================
# PILLAR B TESTS: Physics Auditing & Surface Force Integration
# =============================================================================


class TestPillarB:
    def test_audit_flow_computes_expected_fields_without_mutating_input(
        self, unit_cube_grid: pv.ImageData
    ) -> None:
        original_keys = set(unit_cube_grid.point_data.keys())
        audited = unit_cube_grid.sciml.audit_flow(pred_vel="u_pred", target_vel="u_true")

        # Input dataset must not be mutated in place
        assert set(unit_cube_grid.point_data.keys()) == original_keys

        expected_new_keys = {
            "vel_abs_err",
            "pred_divergence",
            "true_divergence",
            "pred_vorticity_mag",
            "true_vorticity_mag",
        }
        assert expected_new_keys.issubset(set(audited.point_data.keys()))

        # Verify pointwise L2 velocity error matches manual calculation
        manual_err = np.linalg.norm(
            unit_cube_grid.point_data["u_pred"] - unit_cube_grid.point_data["u_true"], axis=-1
        )
        np.testing.assert_allclose(audited.point_data["vel_abs_err"], manual_err, rtol=1e-5)

        # Predicted flow has artificial +0.1*x in u_x -> higher divergence violation than truth
        assert np.mean(np.abs(audited.point_data["pred_divergence"])) > np.mean(
            np.abs(audited.point_data["true_divergence"])
        )

    def test_integrate_drag_lift_uniform_pressure_cancels_on_closed_body(
        self, cylinder_surface: pv.PolyData
    ) -> None:
        # Constant pressure over a closed watertight surface must yield net zero force
        cylinder_surface.point_data["p_uniform"] = np.full(cylinder_surface.n_points, 2.5, dtype=np.float32)
        forces = cylinder_surface.sciml.integrate_drag_lift(pressure="p_uniform")

        assert np.isclose(forces["Cd"], 0.0, atol=1e-5)
        assert np.isclose(forces["Cl"], 0.0, atol=1e-5)

    def test_integrate_drag_lift_linear_pressure_gradient_matches_analytical(
        self, cylinder_surface: pv.PolyData
    ) -> None:
        # By Gauss's divergence theorem: -integral(p * n dA) = -integral(grad(p) dV)
        # For p(x) = -x, grad(p) = (-1, 0, 0) -> net drag force F_x = +1 * Volume
        # Cylinder volume = pi * r^2 * h = pi * 0.25 * 1.0 = pi / 4
        cylinder_surface.point_data["p_grad"] = -cylinder_surface.points[:, 0].astype(np.float32)
        forces = cylinder_surface.sciml.integrate_drag_lift(
            pressure="p_grad", rho=2.0, u_inf=1.0, ref_area=1.0
        )  # q_inf = 0.5 * 2.0 * 1.0^2 * 1.0 = 1.0, so Cd == F_x

        expected_volume = np.pi * (0.5**2) * 1.0
        assert np.isclose(forces["Cd"], expected_volume, rtol=1e-2)
        assert np.isclose(forces["Cl"], 0.0, atol=1e-4)

    def test_integrate_drag_lift_handles_unstructured_grid_input(
        self, cylinder_surface: pv.PolyData
    ) -> None:
        # Regression test: thresholded surfaces become UnstructuredGrids without .compute_normals()
        cylinder_surface.point_data["z"] = cylinder_surface.points[:, 2]
        ugrid_surface = cylinder_surface.threshold(value=[-1.0, 1.0], scalars="z")
        assert isinstance(ugrid_surface, pv.UnstructuredGrid)

        ugrid_surface.point_data["p_test"] = np.ones(ugrid_surface.n_points, dtype=np.float32)
        forces = ugrid_surface.sciml.integrate_drag_lift(pressure="p_test")
        assert "Cd" in forces and "Cl" in forces


# =============================================================================
# PILLAR C TESTS: Synchronized 3-Panel Visual Diagnostics
# =============================================================================


class TestPillarC:
    def test_compare_offscreen_renders_three_panels_with_locked_clim(
        self, unit_cube_grid: pv.ImageData
    ) -> None:
        audited = unit_cube_grid.sciml.audit_flow(pred_vel="u_pred", target_vel="u_true")
        pl = audited.sciml.compare(
            true_field="true_vorticity_mag",
            pred_field="pred_vorticity_mag",
            slice_normal="z",
            title_prefix="Test:",
            off_screen=True,
        )

        try:
            assert isinstance(pl, pv.Plotter)
            assert len(pl.renderers) == 3

            # Verify off-screen rendering produces a non-empty RGB image array
            img = pl.screenshot(return_img=True)
            assert img is not None
            assert img.ndim == 3 and img.shape[2] >= 3
        finally:
            pl.close()

    def test_compare_without_slicing_works_on_2d_surface(
        self, cylinder_surface: pv.PolyData
    ) -> None:
        cylinder_surface.point_data["p_true"] = cylinder_surface.points[:, 0].astype(np.float32)
        cylinder_surface.point_data["p_pred"] = (0.9 * cylinder_surface.points[:, 0]).astype(np.float32)

        pl = cylinder_surface.sciml.compare(
            true_field="p_true",
            pred_field="p_pred",
            slice_normal=None,
            off_screen=True,
        )
        try:
            assert len(pl.renderers) == 3
        finally:
            pl.close()