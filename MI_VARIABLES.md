# Phase × Behavior MI variable guide

Mutual information (MI) measures how much knowing circadian phase reduces uncertainty about behavioral identity. Higher MI means behavioral identity is more strongly associated with circadian phase. MI is not classification accuracy and is not “percent variance explained.” All primary corrected MI values use complete cycles and the structure-preserving circular-shift null.

## Core MI quantities

### `MI_raw_bits`

Definition: observed mutual information between circadian phase and behavioral identity.

Formula: `I(P;B)`

Interpretation: how much knowing phase reduces uncertainty about behavior in the observed data.

Use: detailed/QC value, not the preferred primary genotype-level outcome.

### `MI_null_mean_bits`

Definition: mean MI from the structure-preserving circular-shift null distribution.

Interpretation: expected apparent phase-behavior information when intact daily behavioral sequences are randomly aligned to circadian phase.

### `MI_null_SD_bits`

Definition: standard deviation of the null MI distribution.

Use: QC/diagnostic.

### `MI_excess_bits`

Formula: `MI_excess = MI_observed - mean(MI_null)`

Interpretation: phase-related information above the structure-preserving null expectation.

Use: primary animal-level MI quantity.

### `MI_z`

Formula: `MI_z = (MI_observed - mean(MI_null)) / SD(MI_null)`

Interpretation: how many null-distribution standard deviations the observed MI lies above the null mean.

This is a single-animal diagnostic relative to that animal’s permutation null. It is not the genotype-level statistical test and should not automatically be used as the primary WT-versus-KO comparison variable.

### `H_behavior_bits`

Definition: Shannon entropy of behavioral identity.

Interpretation: baseline uncertainty in which behavior the animal is performing before knowing circadian phase.

Use: interpretive/QC quantity.

### `NMI_raw`

Formula: `NMI_raw = MI_raw / H_behavior`

Interpretation: fraction of behavioral-state uncertainty reduced by knowing circadian phase.

It is not classification accuracy and is not percent variance explained.

### `NMI_null_mean`

Definition: mean normalized MI from the null distribution.

### `NMI_excess`

Definition: `NMI_excess = NMI_observed - mean(NMI_null)`

Interpretation: excess fraction of behavioral uncertainty explained by phase above the null expectation.

Use: interpretive value, not the primary downstream outcome.

## Behavioral decomposition

The full 9-state MI is decomposed into (1) rest-versus-non-rest organization and (2) organization among the 8 non-resting behaviors.

### Full 9-state MI

Behavioral states: eating, drinking, rearing, climbing, digging, nesting, resting, grooming, and locomotion.

Question: “How much does circadian phase tell us about the animal’s behavioral state overall?”

### Rest/non-rest MI

Question: “How much does circadian phase tell us whether the animal is resting or not resting?”

### Conditional 8-state non-rest MI

Behavioral states: eating, drinking, rearing, climbing, digging, nesting, grooming, and locomotion.

Question: “Given that the animal is not resting, how much does circadian phase tell us which behavior it is performing?”

Resting samples are excluded from the contingency table. Remaining samples retain their original circadian phase; time is not concatenated or compressed.

### Weighted conditional 8-state contribution

Formula: `P(non-rest) * MI_conditional8`

Interpretation: contribution of organization within the non-resting behavioral repertoire to total 9-state MI.

The decomposition is:

```text
MI_9state = MI_rest_nonrest + P(non-rest) * MI_conditional8
```

The same relationship holds for `MI_excess`:

```text
MI9_excess = MI_rest_nonrest_excess
             + MI_conditional8_weighted_excess
```

## Rest proportions

### `P_rest`

Fraction of complete-cycle samples classified as resting.

### `P_nonrest`

Fraction of complete-cycle samples classified as non-resting.

`P_rest + P_nonrest = 1`.

## FRP fields

### `FRP_hours`

Free-running period used to convert elapsed time into relative circadian phase.

### `FRP_source`

Possible values:

- `default`: no FRP supplied; 24.0 h used.
- `user_supplied`: FRP supplied with `--frp-hours`.

`FRP_source` is provenance, not a biological outcome.

## Recommended downstream variables

The compact `animal_mi_summary.csv` contains exactly these columns:

- `animal_id`
- `FRP_hours`
- `FRP_source`
- `P_rest`
- `P_nonrest`
- `MI9_excess_bits`
- `MI_rest_nonrest_excess_bits`
- `MI_conditional8_excess_bits`
- `MI_conditional8_weighted_excess_bits`

Recommended interpretation:

- Primary overall MI outcome: `MI9_excess_bits`
- Decomposition/biological interpretation: `MI_rest_nonrest_excess_bits`, `MI_conditional8_excess_bits`, and `MI_conditional8_weighted_excess_bits`
- Behavioral composition: `P_rest` and `P_nonrest`
- FRP: `FRP_hours`
- Provenance: `FRP_source`

Do not assume that every variable should automatically receive a separate statistical test. The detailed `mi_results.csv` retains raw MI, null summaries, MI z-scores, entropy, NMI, weighted values, and other QC information.
