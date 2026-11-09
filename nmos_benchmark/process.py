"""ViennaPS blanket boron threshold-adjust implant before gate/source-drain formation.

This is a reduced process flow, not a complete foundry process. The implant
shape is taken from a public 1e13 cm^-2 table and dose-normalized by ViennaPS.
No dose-dependent damage, TED, or kinetic activation calibration is claimed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "modeldb"


def verify_tables() -> None:
    provenance = json.loads((DB / "provenance.json").read_text())
    for item in provenance["files"]:
        if hashlib.sha256((DB / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Model table checksum mismatch: {item['path']}")


def anneal_parameters() -> dict:
    with (DB / "anneal/annealing.csv").open() as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return {r["parameter"]: (float(r["prefactor"]), float(r["Ea"]))
                for r in rows if r["name"] == "boron"}


def run_process(device: dict, recipe: dict, output: Path) -> dict:
    import viennaps as vps

    verify_tables()
    vps.setDimension(2)
    vps.setNumThreads(1)
    p = device["process"]
    temperature_k = recipe["anneal_temperature_C"] + 273.15
    if not 1000 <= recipe["anneal_temperature_C"] <= 1100:
        raise ValueError("This configured model is restricted to 1000..1100 C.")
    if p["implant_energy_keV"] not in [8, 10, 30, 50, 100, 130, 200]:
        raise ValueError("Use a tabulated implant energy; extrapolation is disabled.")
    dx = p["grid_nm"]
    depth = device["geometry"]["substrate_depth_nm"]
    domain = vps.Domain(dx, 100.0)
    vps.MakePlane(domain, -depth, vps.Material.Si).apply()
    vps.MakePlane(domain, 0.0, vps.Material.Si, True).apply()
    domain.generateCellSet(2 * dx, vps.Material.Air, True, True)
    cells = domain.getCellSet()
    cells.buildNeighborhood()
    profile = vps.ImplantTableModel(
        str(DB / "implant/boron_in_silicon_crystalline.csv"),
        "B", "Si", "crystalline", p["implant_energy_keV"], 7.0, 22.0,
        1e13, 0.0,
    )
    implant = vps.IonImplantation()
    implant.setImplantModel(profile)
    implant.setDose(recipe["implant_dose_cm2"])
    implant.setTiltAngle(7.0)
    implant.setLengthUnit(1e-7)  # cm per nm; fields remain nm^-3
    implant.setDoseControl(vps.ImplantDoseControl.WaferDose)
    implant.setConcentrationLabel("B_total")
    vps.Process(domain, implant, 0.0).apply()
    anneal = vps.Anneal()
    anneal.setSpeciesLabel("B_total")
    anneal.setTemperature(temperature_k)
    anneal.setDuration(recipe["anneal_time_s"])
    pars = anneal_parameters()
    anneal.setArrheniusParameters(*pars["d"])
    anneal.setMode(vps.AnnealMode.GaussSeidel)
    anneal.setImplicitSolverOptions(2000, 1e-8, 1.0)
    anneal.setDiffusionMaterials([vps.Material.Si])
    anneal.setBlockingMaterials([vps.Material.Air])
    anneal.enableSolidActivation(True)
    anneal.setActiveLabel("B_active")
    prefactor_cm3, energy_ev = pars["solubility"]
    anneal.setSolidSolubilityArrhenius(prefactor_cm3 * 1e-21, energy_ev)
    vps.Process(domain, anneal, 0.0).apply()
    centers = np.asarray([cells.getCellCenter(i) for i in range(cells.getNumberOfCells())])
    total = np.asarray(cells.getScalarData("B_total")) * 1e21
    active = np.asarray(cells.getScalarData("B_active")) * 1e21
    # Sample the interior, avoiding lateral edge artifacts. Depth is positive down.
    mask = ((np.abs(centers[:, 0]) < 20) & (centers[:, 1] < 0)
            & (centers[:, 1] > -depth))
    depths, inverse = np.unique(-centers[mask, 1], return_inverse=True)
    counts = np.bincount(inverse)
    total_1d = np.bincount(inverse, weights=total[mask]) / counts
    active_1d = np.bincount(inverse, weights=active[mask]) / counts
    if (len(depths) < 5 or not np.isfinite(active_1d).all()
            or np.min(active_1d) < 0 or not np.any(active_1d > 0)):
        raise RuntimeError("ViennaPS produced an invalid active doping profile.")
    if np.any(active_1d > total_1d * (1 + 1e-6) + 1):
        raise RuntimeError("Active concentration exceeds total concentration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, depth_nm=depths, boron_total_cm3=total_1d,
                        boron_active_cm3=active_1d)
    return {"profile_path": str(output.resolve()),
            "active_peak_cm3": float(active_1d.max()),
            "active_sheet_cm2": float(np.trapezoid(active_1d, depths * 1e-7)),
            "total_sheet_cm2": float(np.trapezoid(total_1d, depths * 1e-7))}
