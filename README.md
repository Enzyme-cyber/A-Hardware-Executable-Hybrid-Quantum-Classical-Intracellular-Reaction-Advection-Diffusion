# A Hardware-Executable Hybrid Quantum–Classical Domain-Decomposition Framework for Three-Dimensional Intracellular Reaction–Advection–Diffusion

This repository contains the final three-dimensional hybrid quantum-classical intracellular transport solver associated with the manuscript. It combines a compact conical-frustum finite-volume model, selected second-order Carleman linearization, LCHS-based quantum evolution, QPanda-based real-QPU execution, and multiplicative Schwarz coupling across 24 subdomains. 

Note: the number of subdomains could be modified for more refined simulations.

## Requirements

Python 3.10 or newer; NumPy, SciPy, Matplotlib, and Plotly; `pyqpanda3` 0.4.x for real-QPU execution; QPanda cloud credentials and access to a supported physical QPU backend for hardware reproduction.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```



## Quick validation without QPU access

Firstly, validate the common 24-subdomain topology and bidirectional schedule:

```bash
python src/validate_bidirectional_schedule.py \
  --config configs/smoke/layer_geometry_smoke.json
```

Then create a geometry-only plan for a formal case:

```bash
python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_1.json \
  --dry-plan
```

Run the reduced end-to-end classical smoke test:

```bash
python src/layer_schwarz_qpu_controller.py \
  --config configs/smoke/layer_geometry_smoke.json \
  --mode classical_mock \
  --max-cones 3 \
  --max-schwarz-iterations 1
```

The three-cone(subdomains) limit keeps this check short while completing the first upper/lower/target triangular cell. It verifies configuration loading, model construction, the per-cone runner, output writing, and directed interface handoffs. It is not intended to reproduce the complete antipodal meeting.

The packaged example configurations are intentionally set to **Carleman order = 2** and **retained Pauli terms = 2** where applicable, so that first-time execution remains practical on current QPU hardware. These values are starter settings rather than fixed limitations of the framework: both can be increased in the configuration files. For quantitative reproduction of the manuscript results, please use the Carleman order and Pauli-term settings reported for the corresponding experiment in the **Supplementary Material**, rather than assuming that the repository defaults are the final scientific settings. Increasing either setting can improve approximation fidelity, but it also increases circuit complexity.



## Real-QPU setup

QPanda is the software and backend-abstraction layer. In the manuscript runs, QPanda-dispatched results were obtained from physical QPU execution (WK_C180_1). The actual backend can be selected through the configuration or environment without changing the numerical framework.

## Setup your QPU API

Real-QPU backend API can be obtained through webportal of Origin Quantum:

[https://console.originqc.com.cn/zh/services](https://console.originqc.com.cn/zh/services)

Where  QPU time is free up to 120s, more shots beyond 120s need to be paid. So ++**we prefer Schwarz Iteration 1 for you to reproduce the results.**++

Windows PowerShell:

```powershell
$env:QPANDA_QCLOUD_API_KEY="YOUR_API_KEY"
$env:QPANDA_QCLOUD_BACKEND="auto"
```

Linux/macOS:

```bash
export QPANDA_QCLOUD_API_KEY="YOUR_API_KEY"
export QPANDA_QCLOUD_BACKEND="auto"
```

Optional connectivity check:

```bash
python src/originq_cloud_setup.py --run-bell
```



## Run in 1 Schwarz Iteration to SAVE YOUR SHOTS

Run each case separately from the repository root. Each configuration writes to a distinct output directory, preventing one case from overwriting another. Run each example within 1 Schwarz iteration would burn you about 22s of QPU chip time and 60s QPU time quota. Total runtime including queuing and waiting time is about 20 mins to few hours depends on the queuing time.

```bash
python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_1.json \
  --mode qpu \
  --max-targets-per-cone 1 \
  --max-schwarz-iterations 1 \
  --resume

python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_side.json \
  --mode qpu \
  --max-targets-per-cone 1 \
  --max-schwarz-iterations 1 \
  --resume

python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_wholeside.json \
  --mode qpu \
  --max-targets-per-cone 1 \
  --max-schwarz-iterations 1 \
  --resume
```

Expected output roots are:

- `outputs/fig6/parameters_1/`
- `outputs/fig6/parameters_side/`
- `outputs/fig6/parameters_wholeside/`



## Reproduce the three formal Fig. 6 cases

Switch the Schwarz-iterations input to 2 to reproduce the three formal Fig. 6 cases. WARN: Schwarz-iterations 2 would burn you about 120~130s per run!! Please mind your free QPU quota was limited within 120s.

```bash
python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_1.json \
  --mode qpu \
  --max-targets-per-cone 0 \
  --max-schwarz-iterations 2 \
  --resume

python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_side.json \
  --mode qpu \
  --max-targets-per-cone 0 \
  --max-schwarz-iterations 2 \
  --resume

python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_wholeside.json \
  --mode qpu \
  --max-targets-per-cone 0 \
  --max-schwarz-iterations 2 \
  --resume
```

`--resume` reuses completed per-cone outputs from schwarz 1. Remove the flag when a completely new run is required. Cloud queue/waiting time is external to the numerical output and is not included in the backend execution time used by the manuscript timing protocol.

## Visualize an output

```bash
python src/layer_geometry_visualizer.py \
  --output-dir outputs/fig6/parameters_1 \
  --source qpu
```

Add `--animate` for a scheduling GIF or `--cone-id L1-C001` for the detailed
single-subdomain report.

## Reproducibility scope

The source code and configurations can be inspected and executed locally. The  
classical smoke test requires no cloud credentials. Exact hardware reproduction  
also depends on provider access, backend availability, calibration state, shot  
count, and queue conditions. QPU credentials are intentionally not included.