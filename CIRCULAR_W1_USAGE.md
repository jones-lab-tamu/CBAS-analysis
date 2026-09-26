# Circular W1 cohort analysis

The analysis script accepts any cohort listed in a metadata CSV. Required
columns are `animal` and `group`; optional `display_order` is numeric and sorts
animals for output. Without `--metadata`, the script uses the original eight
pilot animals. The analysis itself supports any number of animals and group
labels. The separate descriptive figure is designed for exactly two selected
groups.

Example metadata:

```csv
animal,group,display_order
675G,LacZ,1
675H,LacZ,2
675I,LacZ,3
675J,LacZ,4
714D,Bmal1KO,5
714E,Bmal1KO,6
714G,Bmal1KO,7
714H,Bmal1KO,8
```

Run the analysis and then generate the two-group figure (replace the example
paths and group labels with the values for your cohort):

```powershell
python circular_w1_repertoire_analysis.py "C:\path\to\Cohort_Data" --metadata "C:\path\to\cohort_metadata.csv"
python circular_w1_repertoire_figure.py --input-dir "C:\path\to\Cohort_Data\Circular_W1_Repertoire" --metadata "C:\path\to\cohort_metadata.csv" --control-group "LacZ" --experimental-group "Bmal1KO"
```

The figure's group flags must match the metadata labels. The figure generator
currently requires exactly two selected groups; the analysis script does not.

## GUI workflow

Run `python circular_w1_launcher.py`, choose the cohort root, confirm the
detected animals and folder-derived group labels, then select **Check Inputs**.
Group and animal names are read-only and come from the immediate
`<cohort_root>/<group>/<animal>` folder structure. Select **Run Circular W1
Analysis** to create/update `cohort_metadata.csv` and run the existing analysis;
then choose the control and experimental groups and select **Generate Circular
W1 Figure**.

## Input layout

`cohort_root` contains one immediate directory per group, then one immediate
directory per animal. For each metadata row, the analysis locates the animal
at `<cohort_root>/<group>/<animal>`:

```text
Cohort_Data/
  <group>/
    <animal>/
      <animal>_<five-digit-index>_curated_aug_model_outputs.csv
      ...
      FRP_Phase_Output/
        frp_phase_summary.csv
        frp_phase_cycles.csv
```

CBAS source CSVs must be directly in the animal folder, use that animal ID as
their filename prefix, and have consecutive five-digit numeric indices (gaps
are rejected). The frozen phase summary must provide `selected_frp_hours` and
`start_ct_used`. The cycle table must provide `cycle_index`,
`elapsed_start_boundary_hours`, `elapsed_end_boundary_hours`,
`missing_bin_count`, and `full_cycle_flag`. The existing phase loader reads the
selected FRP and CT anchor from the summary and the complete-cycle boundaries
from the cycle table; it checks the period against the cycle durations. This
analysis does not estimate FRP. It uses the frozen per-animal CT anchor (the
pilot animals use CT18) and retains the current requirement for four complete
cycles per animal, with no missing source-file indices. The upstream
`frp_phase_analysis.py` period search is limited to 20–28 h; its `--start-ct`
argument is configurable and defaults to 18.

The analysis writes its CSV results, validation summary, and QC figures to:

```text
<cohort_root>/Circular_W1_Repertoire/
```
