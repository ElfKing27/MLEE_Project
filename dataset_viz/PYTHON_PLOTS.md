# Matplotlib plots from the VS Code terminal

From the MLEE project directory, run:

```powershell
python .\dataset_viz\plot_dataset.py --input outputs_parallel_100 --include-pilots
```

This opens a native Matplotlib window. Remove `--include-pilots` to show only
final `dataset.csv` files. `--input` also accepts a device output directory or
an individual CSV. Pilot and final samples remain separate views.

- Drag the handles on the dose, time, and temperature sliders to choose a range.
  Dose uses log10 units: 12 means 10^12 cm^-2.
- The three recipe plots show dose versus time, dose versus temperature, and
  time versus temperature together. All filters apply to all five plots.
- Change the horizontal axis of the two electrical plots, or choose All, Safe,
  Unsafe, or Unknown to filter points by safety.
- Click a point to see its recipe, threshold voltage, leakage, and safety margin.
- Previous and Next switch devices or phases. Reset clears filters; Reload CSVs
  reads a fresh snapshot and resets filters without restarting the generator.
- Dashed lines mark the configured threshold target and leakage limit.

These controls filter measured simulation results. They do not run new recipes
or change the model, dataset, or benchmark qualification. Missing responses are
omitted from the electrical plots; unknown recipes still appear in recipe space.

If the Python interpreter does not have Matplotlib, install it in that interpreter
with `python -m pip install matplotlib`. The interactive window uses TkAgg.
For a noninteractive PNG without Tk, use:

```powershell
python .\dataset_viz\plot_dataset.py --input outputs_parallel_100 --include-pilots --save visualizations/nmos_plots.png
```
