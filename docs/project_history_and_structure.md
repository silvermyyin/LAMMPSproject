# Project History and Structure

A chronological snapshot of the major structural and organisational changes in **LAMMPS-Gen**.

| Date | Change | Notes |
|------|--------|-------|
| 2024-12 | **v0.1** Initial import | Legacy code & data dumped in a flat repo. |
| 2025-01 | Added baseline CSVs | `data/baseline/{train,val,test}.csv` introduced. |
| 2025-03 | First LLM baseline script | `src/generators/run_baseline_experiment.py` created with hard-coded API key; results dumped in `LAMMPSrun/`. |
| 2025-05-10 | **Re-organisation draft** | Inspired by InputFileSynthesis-master. Proposed new directories. |
| 2025-06-10 | **Automated re-org** | Ran `scripts/reorganize_enhanced.sh` – generated new `data/`, `src/`, `scripts/`, etc. Backed up old content to `data_backup_20250612_003400/`. |
| 2025-06-11 | Added comprehensive READMEs | Every top-level folder now has a documented README (✅ existing, 🔧 planned). |
| 2025-06-11 | Testing & Experiment skeletons | `tests/` and `experiments/` directories with guidance files created. |
| 2025-06-12 | **Data de-duplication** | Merged reference `.in` scripts from backup → `data/real_world/research_scripts/`; removed duplicates. |
| 2025-06-12 | Baseline driver patched | • Parameterised reference dir.<br>• Logs → `results/experiments/dated_runs/`.<br>• Uses env var `OPENAI_API_KEY`. |

## Current Directory Tree (Top-level)
```
FinalProject/
  data/                # canonical datasets & scripts
  docs/                # documentation (this file, overview, API ...)
  experiments/         # ongoing/completed/archived experiment bundles
  results/             # generated inputs, outputs, plots, logs
  scripts/             # CLI & helper pipelines
  src/                 # importable Python packages (generators, model, rag ...)
  tests/               # pytest suites
  configs/             # YAML/JSON configs
  data_backup_*        # immutable snapshots of pre-reorg state
  LAMMPSrun/           # scratch dir for local execution (legacy)
```

### Legacy vs. Enhanced
* Anything under `data_backup_*`, `src/generators/`, `LAMMPSrun/`, or `results/baseline/` is **legacy** and will be pruned once all code paths migrate.

### Naming Conventions
* Experiments → `results/experiments/dated_runs/YYYY-MM-DD_<name>/`
* Plots       → `results/plots/<name>_<metric>.png`
* Logs        → `results/logs/<name>_YYYYMMDD_HHMMSS.log`

---

For finer-grained commit-level history, consult `git log --graph --decorate --oneline`. *File updated 2025-06-12.*

### 🛠 Work Completed (Functional)
The table above lists dates; this section groups achievements by theme so new contributors can see *what is working today*.

| Area | Key Outcomes & Artifacts |
|------|--------------------------|
| **Directory Layout** | • New hierarchy (`data/`, `src/`, `scripts/`, `results/`, `tests/`, `experiments/`, `docs/`).<br>• Legacy snapshots preserved in `data_backup_*`. |
| **Data Migration** | • 1 518 reference *.in* files consolidated → `data/real_world/research_scripts/`.<br>• Duplicates removed (kept highest-quality copy). |
| **Baseline Code Refactor** | • `src/generators/run_baseline_experiment.py` parameterised (`reference_dir`, `num_samples`, `temperature`).<br>• Logs & artefacts now saved in `results/experiments/dated_runs/`.<br>• OpenAI key now read from `OPENAI_API_KEY` env var (no hard-coded secret). |
| **Documentation** | • READMEs for every top-level folder, with ✅/🔧 flags.<br>• Two umbrella docs (this file & `project_overview_enhanced.md`). |
| **Testing & Experiments Skeletons** | • `tests/` scaffold with README & pytest config guide.<br>• `experiments/` scaffold with lifecycle README. |
| **Plots & Metrics** | • Baseline run auto-generates CSV results + histograms (`results/plots/*`). |

*Section added 2025-06-12.* 