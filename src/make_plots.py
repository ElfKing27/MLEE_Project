"""Sanity plots: feasibility map, objective landscape, V_th surface."""
import numpy as np
import matplotlib.pyplot as plt
import process_simulator as ps

frac, best, grid = ps.ground_truth(n=120)
LD, TT, TC, obj, feas, vth = grid
print(f"Infeasible fraction: {frac*100:.1f}%")
print(f"Optimum: dose={best['dose']:.2e}  time={best['time']:.1f}s  "
      f"temp={best['temp_C']:.1f}C  V_th={best['V_th']:.3f}V")

# Slice at the temperature nearest the optimum
ti = np.argmin(np.abs(TC[0, 0, :] - best["temp_C"]))
temp_slice = TC[0, 0, ti]
dose_ax = 10.0 ** LD[:, 0, ti]         # x-axis (dose, log)
time_ax = TT[0, :, ti]                 # y-axis (time)
obj2d   = obj[:, :, ti].T
feas2d  = feas[:, :, ti].T
vth2d   = vth[:, :, ti].T

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (1) Feasibility map
ax[0].pcolormesh(dose_ax, time_ax, feas2d.astype(float),
                 cmap="RdYlGn", shading="auto", vmin=0, vmax=1)
ax[0].set_xscale("log")
ax[0].set_title(f"Feasibility (green=safe)  @ {temp_slice:.0f} C\n"
                f"{frac*100:.0f}% of full space infeasible")
ax[0].set_xlabel("implant dose [cm$^{-2}$]"); ax[0].set_ylabel("anneal time [s]")

# (2) Objective with infeasible masked
obj_masked = np.ma.masked_where(~feas2d, obj2d)
pc = ax[1].pcolormesh(dose_ax, time_ax, obj_masked, cmap="viridis", shading="auto")
ax[1].set_xscale("log")
ax[1].scatter([best["dose"]], [best["time"]], marker="*", s=320,
              c="red", edgecolor="white", zorder=5, label="safe optimum")
ax[1].set_title("Objective (I$_{on}$ proxy), infeasible masked")
ax[1].set_xlabel("implant dose [cm$^{-2}$]"); ax[1].set_ylabel("anneal time [s]")
ax[1].legend(loc="lower right"); fig.colorbar(pc, ax=ax[1])

# (3) V_th surface with spec-window contours
pc2 = ax[2].pcolormesh(dose_ax, time_ax, vth2d, cmap="coolwarm", shading="auto")
cs = ax[2].contour(dose_ax, time_ax, vth2d, levels=[ps.VTH_LO, ps.VTH_HI],
                   colors="k", linewidths=2)
ax[2].clabel(cs, fmt={ps.VTH_LO: "V_lo", ps.VTH_HI: "V_hi"})
ax[2].set_xscale("log")
ax[2].set_title("Threshold voltage V$_{th}$ [V]\n(black = spec-window edges)")
ax[2].set_xlabel("implant dose [cm$^{-2}$]"); ax[2].set_ylabel("anneal time [s]")
fig.colorbar(pc2, ax=ax[2])

fig.tight_layout()
fig.savefig("simulator_sanity.png", dpi=130)
print("saved simulator_sanity.png")
