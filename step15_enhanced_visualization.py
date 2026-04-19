"""
Step 15: Discharge-Vehicle-Anxiety-Charging Causality Analysis
==============================================================
Entry-point shim — delegates to the full implementation.

The comprehensive script lives at:
    coupling_analysis/step15_discharge_anxiety_causality.py

Key changes vs the prior stub:
* SHAP visualization uses a custom bar chart of mean |SHAP| values instead of
  shap.summary_plot(), which caused rendering failures.
* All labels and titles are in English (no Chinese characters).
* Every visualization is wrapped in try/except so failures are reported and
  skipped without stopping the rest of the run.
* Missing data is handled gracefully with informative messages.

Run the full analysis from the repository root:

    python coupling_analysis/step15_discharge_anxiety_causality.py

Outputs (15 figures + CSVs + TXT report) are written to:
    ./coupling_analysis/results/figures_step15/
"""

import runpy
import os

_script = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "coupling_analysis",
    "step15_discharge_anxiety_causality.py",
)

if __name__ == "__main__":
    runpy.run_path(_script, run_name="__main__")
