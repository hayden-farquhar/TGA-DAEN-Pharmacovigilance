# Negative Control Selection Rubric

## Overview

This document describes the decision framework used to curate negative
control drug--adverse event pairs for ML evaluation in the TGA DAEN
pharmacovigilance study. The original negative control set (50
candidates, 20 matched) was audited against prescribing information,
published case reports, pharmacovigilance databases, and pharmacological
plausibility. The audit identified that 15 of 20 matched negatives had
some level of documented evidence and were replaced following the rubric
below.

## Decision Rubric

Each candidate negative control pair was evaluated against four criteria,
following the multi-source concordance approach of Ryan et al. (2013),
Coloma et al. (2013), and the OHDSI framework:

### Criterion 1: Product Label Check

The adverse event must NOT appear in any section of the drug's approved
product label (FDA Prescribing Information, TGA Product Information, or
EU SmPC), including:

- Boxed Warning
- Warnings and Precautions
- Adverse Reactions (clinical trials)
- Adverse Reactions (postmarketing)

**Pairs excluded by this criterion:**

| Drug | Reaction | Label Source |
|------|----------|-------------|
| Naproxen | Agranulocytosis | FDA PI (Hemic and Lymphatic) |
| Diclofenac | Agranulocytosis | FDA PI (Hemic and Lymphatic) |
| Celecoxib | Agranulocytosis | FDA PI (postmarketing) |
| Simvastatin | Stevens-Johnson Syndrome | FDA PI (postmarketing) |
| Ibuprofen | Agranulocytosis | FDA PI (Hemic and Lymphatic) |

### Criterion 2: Published Case Report Search

PubMed and Google Scholar were searched for case reports or case series
linking the drug to the adverse event. Pairs with positive rechallenge,
class-effect documentation, or multiple independent case reports were
excluded.

**Pairs excluded by this criterion:**

| Drug | Reaction | Evidence |
|------|----------|----------|
| Lithium | Rhabdomyolysis | Multiple case reports; indirect mechanism via NDI; Bateman & Larner 1991 |
| Sertraline | Rhabdomyolysis | 889 FAERS cases (SSRI class effect); serotonin syndrome pathway |
| Fluoxetine | Rhabdomyolysis | 216 FAERS cases; serotonin syndrome pathway |
| Clozapine | Pulmonary Fibrosis | 33 cases reported to TGA/MHRA; interstitial lung disease documented |
| Haloperidol | Rhabdomyolysis | NMS pathway; well-documented in antipsychotic class |
| Ciprofloxacin | Agranulocytosis | Fluoroquinolone class effect; case reports exist |
| Carbamazepine | Rhabdomyolysis | Seizure and NMS pathways documented |

### Criterion 3: Pharmacological Plausibility Assessment

The drug's mechanism of action was evaluated for any direct or indirect
pathway to the adverse event, including:

- Direct pharmacological effect
- Indirect pathway (e.g., drug -> dehydration -> rhabdomyolysis)
- Confounding by indication (e.g., RA patients receive both methotrexate
  and have increased thyroid disease risk)
- Drug interaction pathway (e.g., warfarin increasing statin levels ->
  rhabdomyolysis)

**Pairs excluded by this criterion:**

| Drug | Reaction | Pathway |
|------|----------|---------|
| Methotrexate | Central Hypothyroidism | Confounded by RA-thyroid disease overlap; TT4 reduction documented |
| Metformin | Rhabdomyolysis | Lactic acidosis pathway in overdose; FDA label mentions 1 case |
| Warfarin | Rhabdomyolysis | CYP interaction increasing statin levels |
| Amiodarone | Rhabdomyolysis | CYP3A4 interaction with statins; FDA label revision |
| Amiodarone | Agranulocytosis | Direct neutropenia reports + indirect thyrotoxicosis pathway |
| Paclitaxel | Hypothyroidism | Emerging evidence of taxane-induced thyroid dysfunction |
| Ciprofloxacin | Myocarditis | Fluoroquinolone-induced eosinophilic myocarditis documented |

### Criterion 4: Idiosyncratic Reaction Avoidance

Reactions that occur as rare idiosyncratic effects across many drug
classes were avoided as negative reactions entirely, as they carry
inherent risk of misclassification. Following Hauben et al. (2016), who
found 17% contamination in the OMOP reference set, we excluded:

- **Rhabdomyolysis** -- documented with statins, SSRIs, antipsychotics
  (NMS), lithium (NDI), and many other classes
- **Agranulocytosis** -- documented with NSAIDs, antithyroid drugs,
  antipsychotics, some antibiotics, and others
- **Stevens-Johnson Syndrome** -- documented with anticonvulsants,
  antibiotics, allopurinol, and even statins

Instead, the revised set uses mechanism-specific reactions where the
pharmacological pathway is clearly defined:

| Reaction | Specific Mechanism | Safe For |
|----------|-------------------|----------|
| Tendon Rupture | Fluoroquinolone collagen disruption | All non-fluoroquinolones |
| Serotonin Syndrome | 5-HT reuptake inhibition/agonism | Non-serotonergic drugs |
| NMS | Dopamine D2 antagonism | Non-dopaminergic drugs |
| Lactic Acidosis | Mitochondrial Complex I inhibition | Non-biguanides/NRTIs |
| Drug Dependence | Mu-opioid/GABAergic agonism | Non-opioids/sedatives |
| Cardiomyopathy | Anthracycline/trastuzumab cardiotoxicity | Non-cardiotoxic drugs |

## Expanded Negative Controls (February 2026)

A systematic search for additional negative controls surveyed the top 40
drugs in the DAEN and paired them against mechanism-specific reactions
(Serotonin Syndrome, Priapism, Drug Dependence, Pulmonary Fibrosis,
Lactic Acidosis, Cardiomyopathy, Tendon Rupture). Over 55 candidates
were evaluated; the majority (~67%) were rejected due to documented
associations, confounding by indication, or confounding by co-prescription.
Five additional pairs passed all four rubric criteria:

| Drug | Reaction | Reports | EB05 | Rationale |
|------|----------|---------|------|-----------|
| Ibuprofen | Serotonin Syndrome | 4 | 0.79 | No serotonergic mechanism; COX inhibitor only |
| Ceftriaxone | Priapism | 4 | 1.70 | No alpha-adrenergic or serotonergic activity |
| Aspirin | Drug Dependence | 3 | 0.23 | No mu-opioid or GABAergic agonism |
| Atenolol | Pulmonary Fibrosis | 3 | 1.30 | No fibrogenic mechanism; beta-blockers not implicated |
| Metoprolol | Pulmonary Fibrosis | 4 | 1.77 | Same as atenolol; no fibrogenic mechanism |

**Rejected candidates (examples):**

| Drug | Reaction | Reason for Rejection |
|------|----------|---------------------|
| Rosuvastatin | Cardiomyopathy | CoQ10 depletion mechanism documented |
| Ibuprofen | Lactic Acidosis | Overdose context: pyruvate dehydrogenase inhibition |
| Furosemide | Lactic Acidosis | Confounding by metformin co-prescription |
| ACE inhibitors | Cardiomyopathy | Confounding by indication (prescribed for HF) |
| Pantoprazole | Cardiomyopathy | Emerging signal in post-marketing surveillance |
| Gliclazide | Lactic Acidosis | Confounding by metformin co-prescription |
| Tocilizumab | Pulmonary Fibrosis | Confounding by indication (RA-ILD overlap) |

## Revised Reference Set Composition

| Component | Original | Revised (Feb 2026 v1) | Revised (Feb 2026 v2) | Expanded (v3) |
|-----------|----------|-----------------------|-----------------------|---------------|
| Positive candidates | 51 | 51 (unchanged) | 51 (unchanged) | 51 + 99 external |
| Positive matched | 50 | 50 (unchanged) | 50 (unchanged) | 142 |
| Negative candidates | 50 | 73 | 128 | 128 + 26 external |
| Negative matched | 20 | 21 | 26 | 28 |
| Total reference set | 70 | 71 | 76 | 170 |
| Positive:Negative ratio | 2.5:1 | 2.4:1 | 1.9:1 | 5.1:1 |

The v2 set (76 pairs) serves as the primary internal reference set. The
expanded v3 set (170 pairs) incorporates OMOP, EU-ADR, and Harpaz
controls mapped via a 40-entry USAN→INN drug name crosswalk. Of 26
external negative candidates, only 2 passed the 4-criterion rubric
(levetiracetam→Dyspnoea, pantoprazole→Dyspnoea); the remaining 24 were
rejected due to documented associations, broad organ-toxicity outcomes,
or pharmacological plausibility concerns. The 5.1:1 class imbalance in
the expanded set is addressed via class-weighted ML (XGBoost
scale_pos_weight, RF class_weight='balanced') and stratified CV.

## Impact on Results

| Metric | Original (contaminated) | Revised v1 (n=71) | Revised v2 (n=76) | Expanded v3 (n=170) |
|--------|------------------------|--------------------|--------------------|---------------------|
| EBGM Specificity | 0.750 (5 FP) | 1.000 (0/21 FP) | 1.000 (0/26 FP) | 1.000 (0/28 FP) |
| EBGM Spec 95% CI | -- | [0.839, 1.000] | [0.868, 1.000] | [0.877, 1.000] |
| EBGM Sensitivity | -- | 0.980 | 0.980 | 0.859 |
| Consensus Specificity | -- | 1.000 (0/21 FP) | 1.000 (0/26 FP) | 1.000 (0/28 FP) |
| PRR Specificity | -- | 0.905 (2/21 FP) | 0.769 (6/26 FP) | 0.786 (6/28 FP) |
| XGBoost DPA AUC-ROC | 0.940 | 0.990 | 0.990 | 0.964 |
| RF All AUC-ROC | -- | 0.985 | 0.992 | 0.970 |

The expanded set's lower EBGM sensitivity (85.9% vs 98.0%) reflects 31
OMOP positive controls with EBGM below the EB05 ≥ 2 threshold (weak or
sub-threshold signals). The tiered analysis confirms 100% EBGM
sensitivity for moderate (EBGM 5–50) and strong (>50) signals.

The expanded set (v2) strengthens the EBGM specificity CI from [0.839, 1.000]
to [0.868, 1.000] while confirming 100% specificity holds. Notably, 3 of the
5 new negatives (ceftriaxone-Priapism, atenolol-Pulmonary Fibrosis,
metoprolol-Pulmonary Fibrosis) triggered PRR/ROR/BCPNN signals but NOT EBGM,
demonstrating that Bayesian shrinkage correctly suppresses low-count false
positives that frequentist methods flag.

## References

1. Ryan PB, Schuemie MJ, Welebob E, et al. Defining a reference set to
   support methodological research in drug safety. *Drug Saf*.
   2013;36(Suppl 1):S33-S47. PMID: 24166222.

2. Coloma PM, Avillach P, Salvo F, et al. A reference standard for
   evaluation of methods for drug safety signal detection using
   electronic healthcare record databases. *Drug Saf*.
   2013;36(1):13-23. PMID: 23315292.

3. Harpaz R, DuMouchel W, LePendu P, et al. Performance of
   pharmacovigilance signal-detection algorithms for the FDA Adverse
   Event Reporting System. *Clin Pharmacol Ther*.
   2013;93(6):539-546. PMID: 23571771.

4. Harpaz R, Odgers D, Gaskin G, et al. A time-indexed reference
   standard of adverse drug reactions. *Sci Data*. 2014;1:140043.
   PMID: 25632348.

5. Osokogu OU, Fregonese F, Ferrajolo C, et al. Pediatric drug safety
   signal detection: a new drug-event reference set for performance
   testing. *Drug Saf*. 2015;38(2):207-217. PMID: 25663078.

6. Hauben M, Aronson JK, Ferner RE. Evidence of misclassification of
   drug-event associations classified as gold standard 'negative
   controls' by the Observational Medical Outcomes Partnership (OMOP).
   *Drug Saf*. 2016;39(5):421-432. PMID: 26879560.

7. Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool
   for detecting confounding and bias in observational studies.
   *Epidemiology*. 2010;21(3):383-388. PMID: 20335814.

8. Schuemie MJ, Hripcsak G, Ryan PB, et al. Measuring signal detection
   performance: can we trust negative controls and do we need them?
   *Drug Saf*. 2016;39(5):391-394.

9. OHDSI. The Book of OHDSI, Chapter 18: Method Validity.
   https://ohdsi.github.io/TheBookOfOhdsi/MethodValidity.html
