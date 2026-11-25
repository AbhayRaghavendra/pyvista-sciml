# `pyvista-sciml`

A lightweight, framework-agnostic [PyVista](https://pyvista.org/) dataset accessor (`mesh.sciml`) for **Scientific Machine Learning (SciML)** and **Neural PDE Surrogates** on 3D Computational Fluid Dynamics (CFD) data.

`pyvista-sciml` bridges the translation and verification gap between traditional VTK/CFD simulation meshes and deep learning frameworks (JAX/Flax, PyTorch Geometric, NVIDIA Modulus) without introducing heavy framework dependencies or state-caching bugs.

![Pillar C Vorticity Audit](docs/images/pillar_c_vorticity.png)

---

## Core Capabilities

1. **Pillar A — Mesh-to-Graph & Boundary Extraction (`to_graph`, `boundary_mask`)**
   * Extracts deduplicated, bidirectional graph connectivity (`edge_index`) and C-contiguous (`order="C"`) `float32` node coordinates from structured or unstructured VTK meshes.
   * Projects named boundary surface patches (`wall`, `inlet`, `outlet`) onto mesh nodes using scale-invariant spatial tolerances.
![Pillar C Vorticity Audit](docs/images/pillar_c_vorticity.png)
2. **Pillar B — Physics Conservation & Force Auditing (`audit_flow`, `integrate_drag_lift`)**
   * Computes pointwise velocity error alongside VTK spatial derivatives for **mass conservation divergence** ($\nabla \cdot \mathbf{u}$) and **vorticity magnitude** ($\Vert{}\nabla \times \mathbf{u}\Vert{}$).
   * Integrates surface pressure distributions across curved 3D bodies into dimensionless **Drag ($C_D$)** and **Lift ($C_L$)** coefficients.
3. **Pillar C — Standardized 3D Visual Diagnostics (`compare`)**
   * Renders camera-linked 3-panel comparisons (**Ground Truth** vs. **Surrogate Prediction** vs. **Absolute Spatial Error**) with locked color limits (`clim`) so amplitude damping and phase errors cannot be masked by auto-scaled colorbars. 
   

---

## Installation

### From Source (Local Development)

```bash
git clone https://github.com/<your-username>/pyvista-sciml.git
cd pyvista-sciml
pip install -e ".[dev]"
```

### Dependencies

* `numpy >= 1.24`
* `pyvista >= 0.43`
* `scipy >= 1.10`

*(Intentionally free of `jax` and `torch` dependencies—outputs clean C-contiguous NumPy arrays compatible with any ML stack.)*

---

## Quick Start

Importing `sciml_accessor` automatically registers the `.sciml` accessor on all `pyvista.DataSet` objects. Top-level functional mirrors are also exported for static type checkers (`mypy`, `pyright`) and IDE autocomplete.

```python
import numpy as np
import pyvista as pv
import sciml_accessor as sciml  # Registers .sciml on pv.DataSet

# 1. Load a 3D CFD mesh and boundary patches
mesh = pv.read("cylinder_wake.vtu")
patches = {
    1: pv.read("boundaries/cylinder_wall.vtp"),
    2: pv.read("boundaries/inlet.vtp"),
}

# 2. Pillar A: Extract deduplicated graph arrays for GNN / Neural Operator input
graph = mesh.sciml.to_graph(patches=patches)
# -> {"coords": (N, 3) float32, "edge_index": (2, E) int32, "node_type": (N,) int32}

# 3. Attach surrogate model predictions (e.g., from JAX/Flax or PyTorch Geometric)
mesh.point_data["u_pred"] = predicted_velocity  # (N, 3) float32
mesh.point_data["p_pred"] = predicted_pressure  # (N,) float32

# 4. Pillar B: Audit physical conservation laws & integrated aerodynamic forces
audited = mesh.sciml.audit_flow(pred_vel="u_pred", target_vel="u_true")
wall_surf = audited.extract_surface(algorithm="dataset_surface").threshold(
    value=[0.45, 0.68], scalars="r"
)
forces = wall_surf.sciml.integrate_drag_lift(
    pressure="p_pred", rho=1.0, u_inf=1.0, ref_area=1.0
)
print(f"Predicted Cd: {forces['Cd']:.4f}, Cl: {forces['Cl']:.4f}")

# 5. Pillar C: Render camera-linked, color-locked 3-panel comparison
pl = audited.sciml.compare(
    true_field="true_vorticity_mag",
    pred_field="pred_vorticity_mag",
    slice_normal="z",
    title_prefix="Vorticity:",
    off_screen=True,
)
pl.screenshot("vorticity_audit.png")
pl.close()
```

---

## API Reference

### Pillar A: Graph & Boundary Extraction

* **`mesh.sciml.to_graph(patches=None) -> dict[str, np.ndarray]`**
  Extracts deduplicated bidirectional edges from 2D/3D cells (`edge_index` of shape `(2, E)`), C-contiguous `float32` node coordinates (`coords` of shape `(N, 3)`), and integer boundary labels (`node_type` of shape `(N,)`).
* **`mesh.sciml.boundary_mask(patches, rtol=1e-4) -> np.ndarray`**
  Maps a dictionary of `{label_id: patch_dataset}` surfaces onto volume mesh nodes using a bounding-box-scaled k-d tree tolerance. Unmatched nodes default to `0` (interior fluid); higher label IDs take priority on shared corner nodes.

### Pillar B: Physics & Engineering Auditing

* **`mesh.sciml.audit_flow(pred_vel="u_pred", target_vel="u_true") -> pv.DataSet`**
  Returns a shallow copy of the dataset with `vel_abs_err`, `pred_divergence`, `true_divergence`, `pred_vorticity_mag`, and `true_vorticity_mag` attached to `point_data`.
* **`mesh.sciml.integrate_drag_lift(pressure, rho=1.0, u_inf=1.0, ref_area=1.0) -> dict[str, float]`**
  Auto-extracts a `PolyData` surface if given an `UnstructuredGrid`, orients outward cell normals, weights by exact cell area, and integrates surface pressure into `{"Cd": float, "Cl": float}`.

### Pillar C: Visual Diagnostics

* **`mesh.sciml.compare(true_field, pred_field, slice_normal="z", title_prefix="", off_screen=False) -> pv.Plotter`**
  Creates a 3-panel `pv.Plotter` with linked cameras (`pl.link_views()`) and locked `clim` color bounds across Ground Truth and Prediction alongside an Absolute Error panel.

### Functional Mirrors (Type-Checker Friendly)

Every accessor method can also be called directly as a typed module function:

* `sciml_accessor.to_graph(mesh, patches=...)`
* `sciml_accessor.audit_flow(mesh, pred_vel=..., target_vel=...)`
* `sciml_accessor.compare(mesh, true_field=..., pred_field=...)`

---

## Design Principles & Best Practices

* **Offline ETL Only for Training:** Use `mesh.sciml.to_graph()` in an offline preprocessing step to export static topology once to `.zarr` or `.hdf5` before training. Do not call PyVista inside a live GPU `DataLoader` loop.
* **Stateless & Chainable:** Methods never cache internal tensor state on Python attributes and always return shallow `pv.DataSet` copies so native PyVista filters (`.slice()`, `.threshold()`, `.save()`) can be chained safely.

---

## Project Structure & Testing

```text
pyvista-sciml/
├── sciml_demo/
│   ├── sciml_accessor.py       # Core accessor & functional mirrors
│   └── demo_visual.py          # 3D cylinder wake demonstration script
├── tests/
│   └── test_sciml_accessor.py  # Pytest suite covering Pillars A, B, and C
└── README.md
```

Run the test suite with:

```bash
pytest tests/ -v
```

---

## License

MIT License