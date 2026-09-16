# FRP and Phase × Behavior MI Analysis

This workflow estimates each animal's free-running period (FRP) from behavior during DD, uses that FRP to define circadian phase, and calculates phase × behavior mutual information (MI). The cohort wrapper runs the same validated single-animal workflow across multiple animals and exports one scientific-metrics table plus one QC table.

## 1. Input data

Each animal folder should contain the curated CBAS output CSVs matching:

```text
*_#####_curated_aug_model_outputs.csv
```

For example:

```text
Animal_675G\
    783E_00000_curated_aug_model_outputs.csv
    783E_00001_curated_aug_model_outputs.csv
    ...
```

`783E` is only an example; the filename prefix does not have to be `783E`. Files should be consecutive recordings from the same animal. Do not move, rename, or modify raw CBAS CSVs.

## 2. Single-animal FRP analysis

From the repository directory, run:

```powershell
python frp_phase_analysis.py "C:\path\to\animal_folder"
```

The script estimates FRP with Lomb–Scargle and cosinor, assigns CT and biological cycles using the validated default CT anchor, and writes outputs to:

```text
animal_folder\FRP_Phase_Output\
```

The main file is `frp_phase_summary.csv`. Its `lomb_scargle_frp_hours` value is the primary FRP estimate; `cosinor_frp_hours` is the secondary comparison estimate.

## 3. Single-animal Phase × Behavior MI

Run FRP first, then run:

```powershell
python phase_behavior_mutual_information.py "C:\path\to\animal_folder"
```

MI reads the FRP/phase solution from `FRP_Phase_Output` and writes outputs to:

```text
animal_folder\MI_Output\
```

Main files are:

- `animal_mi_summary.csv` — one-row summary of the main animal-level metrics.
- `mi_results.csv` — component-level results, including full 9-state and conditional 8-state rows.
- `mi_sensitivity.csv` — phase-bin-width and phase-origin sensitivity results.
- `run_summary.txt` — coverage, provenance, and validation checks.

Main scientific outputs:

- `MI9_excess_bits`: primary whole-repertoire phase-organization metric.
- `NMI9_excess`: normalized companion to the whole-repertoire metric.
- `MI_conditional8_excess_bits`: primary non-rest repertoire metric.
- `NMI_excess` on the `conditional_8state_nonrest` row in `mi_results.csv`: normalized companion for the non-rest conditional analysis.
- `MI_rest_nonrest_excess_bits`: phase information carried by broad rest/non-rest organization.
- `MI_conditional8_weighted_excess_bits`: weighted non-rest contribution to the full 9-state decomposition.
- `P_rest` and `P_nonrest`: proportions of classified samples assigned to rest and non-rest behavior.

The MI z-score is a validation/statistical diagnostic, not the primary biological endpoint.

## 4. Cohort / batch analysis

For multiple animals arranged under group folders, use:

```powershell
python cohort_frp_mi_analysis.py "C:\path\to\Cohort_Data"
```

Example structure:

```text
Cohort_Data\
    LacZ\
        675G\
        675H\
    Bmal1KO\
        714D\
        714E\
```

The wrapper recursively finds folders containing the validated CBAS filename pattern, reruns FRP and MI fresh for every animal, and continues if one animal fails. In root-folder mode, `Group` is taken from the animal folder's immediate parent (`LacZ`, `Bmal1KO`, and so on).

Outputs are written to:

```text
Cohort_Data\FRP_MI_Cohort_Output\
```

The main files are `cohort_frp_mi_metrics.csv`, `cohort_frp_mi_qc.csv`, and `cohort_run_summary.txt`.

## 5. Main cohort output

`cohort_frp_mi_metrics.csv` is the downstream biological-outcome table. It contains successful animals only and intentionally excludes QC bookkeeping.

Columns:

```text
Group
Animal
FRP_LS_h
FRP_cosinor_h
MI9_excess_bits
NMI9_excess
MI8_conditional_excess_bits
NMI8_conditional_excess
MI_rest_nonrest_excess_bits
MI8_weighted_excess_bits
P_rest
P_nonrest
```

## 6. QC output

`cohort_frp_mi_qc.csv` contains processing and validation information for every discovered animal, including failures. It includes source-file count, missing indices, recording duration, reported and computational FRP, complete-cycle coverage, partial-cycle exclusions, null and decomposition checks, phase-origin sensitivity, stage status, overall status, and an error message when needed.

Use the QC file to verify that an animal processed correctly. Do not treat QC fields as primary biological outcomes.

## 7. Explicit animal-folder mode

The cohort wrapper can process only the folders supplied explicitly:

```powershell
python cohort_frp_mi_analysis.py "C:\path\to\675G" "C:\path\to\675H"
```

Only those animals are processed. `Group` is left blank in this mode; it is not inferred from arbitrary parent folder names.

## 8. Important analysis assumptions

- The current validated workflow is intended for data collected in DD.
- The existing default CT anchor is part of the validated analysis (CT at elapsed time zero defaults to 18.0).
- Each animal's FRP is used to normalize its circadian phase.
- MI uses complete biological circadian cycles for the corrected analysis.
- Raw CBAS CSVs are read only and are never modified.
- Do not casually change FRP search bounds, phase-bin settings, null structure, permutation count, or other analysis constants.
- If a metric unexpectedly changes from a prior validated result, diagnose the cause before proceeding.

## 9. Recommended workflow

For one animal:

1. Run `frp_phase_analysis.py`.
2. Run `phase_behavior_mutual_information.py`.
3. Check `FRP_Phase_Output`, `MI_Output`, and especially `MI_Output\run_summary.txt`.

For a cohort:

1. Arrange animal folders under their group folders.
2. Run `cohort_frp_mi_analysis.py` on the cohort root.
3. Check `FRP_MI_Cohort_Output\cohort_frp_mi_qc.csv`.
4. Use `FRP_MI_Cohort_Output\cohort_frp_mi_metrics.csv` for downstream analysis.
