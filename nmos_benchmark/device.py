"""Reduced classical planar nMOS drift-diffusion evaluator in DEVSIM.

cm-based 2D cross section; currents are converted from A/cm to A/um.
Prescribed S/D and body contacts are formed AFTER the blanket channel implant.
No calibrated damage leakage, gate tunneling, quantum effects or yield model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

_DLL_HANDLES = []


def load_devsim():
    # Local process environment only. Do not change machine settings or rename DLLs.
    if sys.platform == "win32":
        folder = Path(sys.prefix) / "Library" / "bin"
        if folder.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
            if not os.environ.get("DEVSIM_MATH_LIBS"):
                for name in ("mkl_rt.2.dll", "mkl_rt.3.dll", "mkl_rt.dll"):
                    if (folder / name).is_file():
                        os.environ["DEVSIM_MATH_LIBS"] = str(folder / name)
                        break
    import devsim
    return devsim


def run_device(config: dict, profile_path: Path, curve_path: Path) -> dict:
    ds = load_devsim()
    from devsim.python_packages import simple_physics as sp
    from devsim.python_packages.model_create import CreateSolution

    ds.reset_devsim()
    # Leakage entails nearly cancelling fluxes; ordinary double-precision model
    # evaluation stalled for two reference structures in the integration checks.
    for setting in ("extended_model", "extended_equation", "extended_solver"):
        ds.set_parameter(name=setting, value=True)
    mesh, device = "mesh", "nmos"
    g, fixed, electrical = config["geometry"], config["fixed"], config["electrical"]
    scale = 1e-7  # nm -> cm
    extension = g["source_drain_extension_nm"]
    left, right = extension, extension + g["gate_length_nm"]
    xmax = right + extension
    depth, tox = g["substrate_depth_nm"], g["oxide_thickness_nm"]
    refine = g.get("mesh_refinement", 1)
    # Nonuniform y mesh resolves oxide, surface channel and implanted layer.
    x = np.unique(np.r_[np.linspace(0, left, 9 * refine + 1),
                        np.linspace(left, right, 25 * refine + 1),
                        np.linspace(right, xmax, 9 * refine + 1)])
    y_bulk = np.unique(np.r_[0, np.geomspace(1 / refine, depth, 39 * refine)])
    y = np.unique(np.r_[-tox - 5, np.linspace(-tox, 0, 4 * refine + 1), y_bulk, depth + 5])
    ds.create_2d_mesh(mesh=mesh)
    for axis, values in (("x", x), ("y", y)):
        for index, value in enumerate(values):
            neighbors = np.diff(values[max(0, index - 1):min(len(values), index + 2)])
            spacing = float(neighbors.min()) * scale
            ds.add_2d_mesh_line(mesh=mesh, dir=axis, pos=float(value) * scale, ps=spacing)
    # Native DEVSIM meshing requires a region on both sides of each contact.
    # Ambient has no electrical equations and imposes no conducting path.
    ds.add_2d_region(mesh=mesh, region="ambient", material="Air")
    ds.add_2d_region(mesh=mesh, region="oxide", material="Oxide",
                     xl=left * scale, xh=right * scale, yl=-tox * scale, yh=0)
    ds.add_2d_region(mesh=mesh, region="bulk", material="Silicon",
                     xl=0, xh=xmax * scale, yl=0, yh=depth * scale)
    ds.add_2d_interface(mesh=mesh, name="bulk_oxide", region0="bulk", region1="oxide",
                       xl=left * scale, xh=right * scale, yl=0, yh=0)
    for name, xl, xh, yy, region in (
        ("source", 0, left - 50, 0, "bulk"),
        ("drain", right + 50, xmax, 0, "bulk"),
        ("body", 0, xmax, depth, "bulk"),
        ("gate", left, right, -tox, "oxide"),
    ):
        ds.add_2d_contact(mesh=mesh, name=name, material="metal", region=region,
                          xl=xl * scale, xh=xh * scale, yl=yy * scale, yh=yy * scale)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    xx = np.asarray(ds.get_node_model_values(device=device, region="bulk", name="x")) / scale
    yy = np.asarray(ds.get_node_model_values(device=device, region="bulk", name="y")) / scale
    with np.load(profile_path) as profile:
        implant = np.interp(yy, profile["depth_nm"], profile["boron_active_cm3"],
                            right=0.0)
    sd_depth, lateral, vertical = (fixed[k] for k in
        ("source_drain_depth_nm", "source_drain_lateral_nm", "source_drain_vertical_nm"))
    donors = (fixed["source_drain_doping_cm3"] * 0.25
              * (erfc((xx - left) / lateral) + erfc(-(xx - right) / lateral))
              * erfc((yy - sd_depth) / vertical) + 1)
    acceptors = (fixed["substrate_doping_cm3"] + implant
                 + fixed["body_contact_doping_cm3"] * 0.5
                 * erfc((depth - 20 - yy) / 10))
    for name, values in (("Donors", donors), ("Acceptors", acceptors)):
        ds.node_solution(device=device, region="bulk", name=name)
        ds.set_node_values(device=device, region="bulk", name=name, values=values.tolist())
    ds.node_model(device=device, region="bulk", name="NetDoping", equation="Donors-Acceptors")
    sp.SetSiliconParameters(device, "bulk", 300.0)
    sp.SetOxideParameters(device, "oxide", 300.0)
    sp.CreateSiliconPotentialOnly(device, "bulk")
    sp.CreateOxidePotentialOnly(device, "oxide", "log_damp")
    # Potential reference is intrinsic silicon. Ideal metal gate, no fixed oxide charge.
    gate_offset = 4.05 + 1.12 / 2 - fixed["gate_work_function_eV"]
    for name in ("source", "drain", "body", "gate"):
        ds.set_parameter(device=device, name=sp.GetContactBiasName(name),
                         value=gate_offset if name == "gate" else 0)
        if name == "gate":
            sp.CreateOxideContact(device, "oxide", name)
        else:
            sp.CreateSiliconPotentialOnlyContact(device, "bulk", name)
    sp.CreateSiliconOxideInterface(device, "bulk_oxide")
    ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-10, maximum_iterations=100)
    for solution, initial in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        CreateSolution(device, "bulk", solution)
        ds.set_node_values(device=device, region="bulk", name=solution, init_from=initial)
    sp.CreateSiliconDriftDiffusion(device, "bulk")
    for name in ("source", "drain", "body"):
        sp.CreateSiliconDriftDiffusionAtContact(device, "bulk", name)

    dc_solves = 1  # includes the equilibrium potential-only solve

    def solve():
        nonlocal dc_solves
        dc_solves += 1
        ds.solve(type="dc", absolute_error=1e12,
                 relative_error=electrical.get("relative_tolerance", 1e-5), maximum_iterations=100)

    biases = {"gate": 0.0, "drain": 0.0}

    def ramp(contact, target):
        start = biases[contact]
        for value in np.linspace(start, target, max(2, int(abs(target - start) / 0.05) + 2))[1:]:
            ds.set_parameter(device=device, name=sp.GetContactBiasName(contact),
                             value=float(value) + (gate_offset if contact == "gate" else 0))
            solve()
        biases[contact] = target

    def current(contact):
        return sum(ds.get_contact_current(device=device, contact=contact, equation=eq)
                   for eq in ("ElectronContinuityEquation", "HoleContinuityEquation")) * 1e-4

    solve()
    ramp("drain", electrical["off_drain_voltage_V"])
    currents = np.array([current(c) for c in ("drain", "source", "body")])
    conservation = abs(currents.sum()) / max(np.abs(currents).max(), 1e-18)
    if abs(currents.sum()) > max(1e-17, 1e-3 * np.abs(currents).max()):
        raise RuntimeError(f"Off-state terminal-current imbalance: {conservation:g}")
    off = abs(float(currents[0]))
    ramp("drain", electrical["threshold_drain_voltage_V"])
    gates = np.linspace(0.0, electrical["gate_sweep_max_V"], electrical["gate_sweep_points"])
    ids = []
    criterion = electrical["threshold_current_A_per_um_times_L_um"] / (g["gate_length_nm"] / 1000)
    for vg in gates:
        ramp("gate", float(vg))
        ids.append(abs(current("drain")))
        if len(ids) >= 2 and ids[-2] < criterion <= ids[-1]:
            break
    ids = np.asarray(ids)
    gates = gates[:len(ids)]
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(curve_path, np.c_[gates, ids], delimiter=",",
               header="gate_voltage_V,drain_current_A_per_um", comments="")
    crossings = np.flatnonzero((ids[:-1] < criterion) & (ids[1:] >= criterion))
    vth = None
    if len(crossings):
        i = int(crossings[0])
        # Constant-current threshold with log-current interpolation.
        fraction = ((np.log(criterion) - np.log(max(ids[i], 1e-300))) /
                    (np.log(max(ids[i + 1], 1e-300)) - np.log(max(ids[i], 1e-300))))
        vth = float(gates[i] + fraction * (gates[i + 1] - gates[i]))
    margin = float(np.log(electrical["leakage_limit_A_per_um"] / max(off, 1e-300)))
    return {"vth_V": vth, "ioff_A_per_um": off, "safety_margin": margin,
            "constraint_violated": margin < 0,
            "objective": None if vth is None else -abs(vth - electrical["target_vth_V"]),
            "extraction_status": "ok" if vth is not None else "threshold_not_bracketed",
            "current_balance_relative": float(conservation),
            "iv_curve_path": str(curve_path.resolve()),
            "device_dc_solve_calls": dc_solves,
            "yield_estimate": None}
