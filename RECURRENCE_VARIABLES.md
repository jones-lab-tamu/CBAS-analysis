# Behavioral recurrence variable guide

Recurrence asks whether an animal occupies the same dominant behavioral state in the same normalized circadian phase bin on adjacent complete free-running cycles. The null independently circularly shifts each intact binned cycle, including its validity mask, to disrupt cross-cycle phase alignment while preserving within-cycle structure.

## Core recurrence quantities

For each recurrence component:

- `Recurrence_Raw`: observed recurrence, the fraction of eligible adjacent-cycle same-phase comparisons that match.
- `Recurrence_Null_Mean`: mean recurrence across 10,000 circular-shift permutations.
- `Recurrence_Null_SD`: sample standard deviation of those null recurrence values.
- `Recurrence_Excess`: `Recurrence_Raw - Recurrence_Null_Mean`. This is the primary null-corrected recurrence measure.
- `Recurrence_Normalized`: `(Recurrence_Raw - Recurrence_Null_Mean) / (1 - Recurrence_Null_Mean)`. Zero means no excess above the null expectation, one means perfect recurrence, and negative values mean recurrence below the null expectation. It is `NaN` when the denominator is numerically zero.
- `usable fraction`: usable comparisons divided by all possible adjacent-cycle same-phase comparisons. It reports the amount of data supporting the recurrence estimate; it is not a recurrence value.

If a permutation has no eligible comparisons for a component, that permutation's recurrence is undefined (`NaN`) and is excluded from that component's null mean and sample SD; no value is imputed.

## Recurrence components

- **Full 9-state recurrence** (`full_9state`): exact matching among eating, drinking, rearing, climbing, digging, nesting, resting, grooming, and locomotion.
- **Rest/non-rest recurrence** (`rest_vs_nonrest`): binary matching from the resting count versus the summed count of all eight non-resting behaviors. A bin is resting when resting is larger, non-resting when the summed non-resting count is larger, and unusable on an exact binary tie.
- **Conditional 8-state non-rest recurrence** (`conditional_8state_nonrest`): restricted to bins classified as non-resting by that binary rule, then assigned by the dominant behavior among the eight non-resting counts. Exact ties among those eight behaviors are unusable.

The conditional denominator differs because it requires both binary states to be valid and non-resting and both conditional identities to be usable, excluding resting bins and component-specific ties. Its `n_possible_comparisons` still counts all adjacent-cycle same-phase pairs before those conditions; its usable fraction uses the comparisons that actually enter the conditional calculation. Resting bins are not removed or time-compressed before pairing.

The three representations are derived independently from the same underlying per-bin behavior counts; the rest/non-rest and conditional representations are not reconstructed from the 9-state winner. Each component has its own validity and tie rules, while sharing the total valid-coverage threshold.

These three measures are related views of the same binned behavior but are not additive. The conditional measure is calculated on a changing non-rest subset, and recurrence is a match probability rather than an information decomposition. Do not construct an additive decomposition between them.

## Intended downstream variables

Use the primary excess variables in `animal_recurrence_summary.csv` for downstream animal-level statistics:

- `Recurrence9_Excess`
- `Recurrence_RestNonrest_Excess`
- `Recurrence8_Conditional_Excess`
