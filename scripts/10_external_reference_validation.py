"""
Script 10: External Reference Set Validation

Maps three external pharmacovigilance reference sets to the DAEN and evaluates
DPA performance against each:

  1. OMOP (Ryan et al. 2013): 399 pairs, 4 outcomes
  2. EU-ADR (Coloma et al. 2013): 93 pairs, 10 outcomes
  3. Harpaz 2014 (time-indexed): 137 pairs, 38 event concepts

Depends on outputs from scripts 03 (DPA results).

Outputs:
    outputs/revisions/revision_omop_validation.csv
    outputs/revisions/revision_euadr_validation.csv
    outputs/revisions/revision_harpaz_validation.csv
    outputs/revisions/revision_external_validation_summary.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_DIR / "data" / "reference"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
REVISION_DIR = PROJECT_DIR / "outputs" / "revisions"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial 95% CI."""
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


# ── Drug name mapping (US → AU conventions) ──────────────────────────────────

DRUG_NAME_MAP = {
    # US/INN generic → Australian (DAEN) active ingredient name
    # This crosswalk addresses USAN vs INN naming differences, which account
    # for most unmatched drugs between OMOP/EU-ADR reference sets and the DAEN.
    #
    # Major USAN-INN differences
    "acetaminophen": "paracetamol",
    "albuterol": "salbutamol",
    "meperidine": "pethidine",
    "epinephrine": "adrenaline",
    "norepinephrine": "noradrenaline",
    "mesalamine": "mesalazine",
    "methimazole": "thiamazole",
    "dalfampridine": "fampridine",
    "glyburide": "glibenclamide",
    "isoproterenol": "isoprenaline",
    #
    # INN spelling variants (US -y- vs international -i-)
    "acyclovir": "aciclovir",
    "valacyclovir": "valaciclovir",
    "ganciclovir": "ganciclovir",
    #
    # US vs British/Australian spelling
    "cyclosporine": "ciclosporin",
    "rifampin": "rifampicin",
    "scopolamine": "hyoscine",
    "nitroglycerin": "glyceryl trinitrate",
    "niacin": "nicotinic acid",
    "penicillin v": "phenoxymethylpenicillin",
    "thioguanine": "tioguanine",
    "lidocaine": "lignocaine",
    "sulfasalazine": "sulfasalazine",
    #
    # Insulin naming (generic class → match as substring)
    "regular insulin human": "insulin",
    "insulin glargine": "insulin glargine",
    #
    # Salt form / formulation variations
    "levothyroxine": "levothyroxine",
    "dextroamphetamine": "dexamfetamine",
    "amphetamine": "amfetamine",
    "cephalexin": "cefalexin",
    "cefuroxime": "cefuroxime",
    #
    # Kept as-is (no name difference but included for completeness)
    "furosemide": "furosemide",
    "rofecoxib": "rofecoxib",
    "rosiglitazone": "rosiglitazone",
    "pioglitazone": "pioglitazone",
    "pantoprazole": "pantoprazole",
    "sunitinib": "sunitinib",
    "lacosamide": "lacosamide",
    "solifenacin": "solifenacin",
    "fidaxomicin": "fidaxomicin",
    "cyclobenzaprine": "cyclobenzaprine",
    "benzonatate": "benzonatate",
}


def normalise_drug(name):
    """Normalise drug name to match DAEN conventions."""
    name = name.strip().lower()
    return DRUG_NAME_MAP.get(name, name)


# ── Outcome concept to MedDRA PT mapping ─────────────────────────────────────

# OMOP outcomes → MedDRA PTs in DAEN
OMOP_OUTCOME_MAP = {
    "OMOP Acute Liver Failure 1": [
        "hepatic failure", "hepatic necrosis", "liver failure",
        "acute hepatic failure", "hepatorenal syndrome",
        "hepatic failure acute", "hepatotoxicity", "hepatitis",
        "hepatitis acute", "drug-induced liver injury",
        "liver injury", "hepatocellular injury",
    ],
    "OMOP Acute Renal Failure 1": [
        "renal failure", "acute kidney injury", "renal impairment",
        "renal failure acute", "anuria", "oliguria",
        "blood creatinine increased", "renal tubular necrosis",
        "nephritis tubulointerstitial", "renal disorder",
    ],
    "OMOP Acute Myocardial Infarction 1": [
        "myocardial infarction", "acute myocardial infarction",
        "cardiac arrest", "coronary artery occlusion",
        "acute coronary syndrome", "troponin increased",
        "st segment elevation",
    ],
    "HOI Upper GI #3": [
        "gastrointestinal haemorrhage", "upper gastrointestinal haemorrhage",
        "melaena", "haematemesis", "gastric haemorrhage",
        "gastric ulcer haemorrhage", "duodenal ulcer haemorrhage",
        "gastrointestinal ulcer haemorrhage", "oesophageal haemorrhage",
    ],
}

# EU-ADR additional outcomes
EUADR_OUTCOME_MAP = {
    **OMOP_OUTCOME_MAP,
    "OMOP Aplastic Anemia 1": [
        "aplastic anaemia", "pancytopenia", "bone marrow failure",
    ],
    "OMOP Anaphylaxis 1": [
        "anaphylactic reaction", "anaphylactic shock", "anaphylaxis",
    ],
    "OMOP SJS 1": [
        "stevens-johnson syndrome", "toxic epidermal necrolysis",
    ],
    "OMOP Leukopenia 1": [
        "neutropenia", "leukopenia", "agranulocytosis",
        "febrile neutropenia", "granulocytopenia",
    ],
    "OMOP Rhabdomyolysis 1": [
        "rhabdomyolysis",
    ],
    "OMOP Cardiac Valve 1": [
        "cardiac valve fibrosis", "heart valve disease",
        "mitral valve incompetence", "aortic valve stenosis",
    ],
}


def match_reference_to_daen(ref_df, outcome_map, disp_df, drug_col, outcome_col,
                             truth_col):
    """
    Match a reference set to the DAEN DPA results.

    For each reference pair (drug, outcome_concept):
      - Normalise drug name
      - Expand outcome concept to MedDRA PTs via outcome_map
      - Check if drug × any PT exists in disp_df
      - Take the strongest signal (highest EBGM) if multiple PTs match

    Returns matched DataFrame with DPA results.
    """
    daen_drugs = set(disp_df["active_ingredient"].unique())
    daen_reactions = set(disp_df["reaction"].str.lower().unique())

    rows = []
    for _, r in ref_df.iterrows():
        drug = normalise_drug(r[drug_col])
        outcome = r[outcome_col]
        truth = int(r[truth_col])

        # Get MedDRA PTs for this outcome
        pts = outcome_map.get(outcome, [outcome.lower()])

        # Find drug in DAEN (substring match)
        drug_matches = [d for d in daen_drugs if drug in d]

        # Find best matching pair
        best = None
        for dm in drug_matches:
            for pt in pts:
                pt_matches = [rxn for rxn in daen_reactions if pt in rxn]
                for pm in pt_matches:
                    match = disp_df[
                        (disp_df["active_ingredient"] == dm)
                        & (disp_df["reaction"].str.lower() == pm)
                    ]
                    if len(match) > 0:
                        candidate = match.iloc[0]
                        if best is None or candidate["ebgm"] > best["ebgm"]:
                            best = candidate

        if best is not None:
            rows.append({
                "ref_drug": r[drug_col],
                "ref_outcome": outcome,
                "ground_truth": truth,
                "matched_drug": best["active_ingredient"],
                "matched_reaction": best["reaction"],
                "n_reports": int(best["a"]),
                "ebgm": best["ebgm"],
                "eb05": best["eb05"],
                "ic025": best["ic025"],
                "signal_prr": best["signal_prr"],
                "signal_ror": best["signal_ror"],
                "signal_ebgm": best["signal_ebgm"],
                "signal_bcpnn": best["signal_bcpnn"],
                "n_methods": int(best["n_methods_signal"]),
                "consensus": best["n_methods_signal"] == 4,
                "matched": True,
            })
        else:
            rows.append({
                "ref_drug": r[drug_col],
                "ref_outcome": outcome,
                "ground_truth": truth,
                "matched_drug": None,
                "matched_reaction": None,
                "n_reports": 0,
                "ebgm": None,
                "eb05": None,
                "ic025": None,
                "signal_prr": None,
                "signal_ror": None,
                "signal_ebgm": None,
                "signal_bcpnn": None,
                "n_methods": None,
                "consensus": None,
                "matched": False,
            })

    return pd.DataFrame(rows)


def evaluate_performance(matched_df, name, out):
    """Evaluate DPA performance on matched reference pairs."""
    matched = matched_df[matched_df["matched"]].copy()
    n_total = len(matched_df)
    n_matched = len(matched)
    n_pos = (matched_df["ground_truth"] == 1).sum()
    n_neg = (matched_df["ground_truth"] == 0).sum()
    n_pos_matched = ((matched["ground_truth"] == 1)).sum()
    n_neg_matched = ((matched["ground_truth"] == 0)).sum()

    out.write(f"\n  {name}:\n")
    out.write(f"    Total pairs: {n_total} ({n_pos} positive, {n_neg} negative)\n")
    out.write(f"    Matched to DAEN: {n_matched}/{n_total} "
              f"({n_pos_matched} pos, {n_neg_matched} neg)\n")

    if n_pos_matched < 3 or n_neg_matched < 3:
        out.write(f"    Too few matched pairs for meaningful evaluation\n")
        return {"name": name, "n_total": n_total, "n_matched": n_matched,
                "n_pos": n_pos_matched, "n_neg": n_neg_matched,
                "sens": None, "spec": None}

    # Evaluate each DPA method
    results = {}
    for method, col in [("PRR", "signal_prr"), ("ROR", "signal_ror"),
                         ("EBGM", "signal_ebgm"), ("BCPNN", "signal_bcpnn"),
                         ("Consensus", "consensus")]:
        tp = ((matched["ground_truth"] == 1) & (matched[col] == True)).sum()
        fn = ((matched["ground_truth"] == 1) & (matched[col] != True)).sum()
        fp = ((matched["ground_truth"] == 0) & (matched[col] == True)).sum()
        tn = ((matched["ground_truth"] == 0) & (matched[col] != True)).sum()

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        sens_ci = _clopper_pearson(tp, tp + fn)
        spec_ci = _clopper_pearson(tn, tn + fp)

        out.write(f"    {method}: Sens={sens:.3f} [{sens_ci[0]:.3f}-{sens_ci[1]:.3f}], "
                  f"Spec={spec:.3f} [{spec_ci[0]:.3f}-{spec_ci[1]:.3f}] "
                  f"(TP={tp}, FN={fn}, FP={fp}, TN={tn})\n")

        results[f"{method}_sens"] = round(sens, 4)
        results[f"{method}_spec"] = round(spec, 4)

    results.update({
        "name": name, "n_total": n_total, "n_matched": n_matched,
        "n_pos": n_pos_matched, "n_neg": n_neg_matched,
    })
    return results


def main():
    import io
    import time

    t0 = time.time()
    print("=" * 70)
    print("  TGA DAEN — External Reference Set Validation")
    print("=" * 70)

    report = io.StringIO()

    class TeeWriter:
        def __init__(self, buf):
            self.buf = buf
        def write(self, s):
            print(s, end="")
            self.buf.write(s)

    out = TeeWriter(report)

    # Load DPA results
    out.write("\nLoading DPA results...\n")
    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")
    out.write(f"  {len(disp_df):,} DPA pairs loaded\n")

    summary_rows = []

    # ── 1. OMOP Reference Set ────────────────────────────────────────────
    out.write("\n" + "=" * 70 + "\n")
    out.write("  1. OMOP Reference Set (Ryan et al. 2013)\n")
    out.write("=" * 70 + "\n")

    omop = pd.read_csv(REFERENCE_DIR / "omop_reference_set.csv")
    out.write(f"  Loaded: {len(omop)} pairs ({(omop['groundTruth']==1).sum()} pos, "
              f"{(omop['groundTruth']==0).sum()} neg)\n")

    omop_matched = match_reference_to_daen(
        omop, OMOP_OUTCOME_MAP, disp_df,
        drug_col="exposureName", outcome_col="outcomeName",
        truth_col="groundTruth")
    omop_matched.to_csv(REVISION_DIR / "revision_omop_validation.csv", index=False)

    result = evaluate_performance(omop_matched, "OMOP", out)
    summary_rows.append(result)

    # ── 2. EU-ADR Reference Set ──────────────────────────────────────────
    out.write("\n" + "=" * 70 + "\n")
    out.write("  2. EU-ADR Reference Set (Coloma et al. 2013)\n")
    out.write("=" * 70 + "\n")

    euadr = pd.read_csv(REFERENCE_DIR / "euadr_reference_set.csv")
    out.write(f"  Loaded: {len(euadr)} pairs ({(euadr['groundTruth']==1).sum()} pos, "
              f"{(euadr['groundTruth']==0).sum()} neg)\n")

    euadr_matched = match_reference_to_daen(
        euadr, EUADR_OUTCOME_MAP, disp_df,
        drug_col="exposureName", outcome_col="outcomeName",
        truth_col="groundTruth")
    euadr_matched.to_csv(REVISION_DIR / "revision_euadr_validation.csv", index=False)

    result = evaluate_performance(euadr_matched, "EU-ADR", out)
    summary_rows.append(result)

    # ── 3. Harpaz 2014 Reference Set ─────────────────────────────────────
    out.write("\n" + "=" * 70 + "\n")
    out.write("  3. Harpaz 2014 Time-Indexed Reference Set\n")
    out.write("=" * 70 + "\n")

    harpaz = pd.read_csv(REFERENCE_DIR / "harpaz_2014_reference_set.csv")
    event_defs = pd.read_csv(REFERENCE_DIR / "harpaz_2014_event_definitions.csv")

    out.write(f"  Loaded: {len(harpaz)} pairs "
              f"({(harpaz['GROUND_TRUTH']==1).sum()} pos, "
              f"{(harpaz['GROUND_TRUTH']==0).sum()} neg)\n")

    # Build Harpaz outcome map from event definitions
    harpaz_outcome_map = {}
    for concept_name, group in event_defs.groupby("EVENT_CONCEPT_NAME"):
        # Use narrow definitions preferentially
        pts = group["MEDDRA_PT"].dropna().str.strip().str.lower().tolist()
        harpaz_outcome_map[concept_name] = pts if pts else [concept_name.lower()]

    harpaz_matched = match_reference_to_daen(
        harpaz, harpaz_outcome_map, disp_df,
        drug_col="DRUG_CONCEPT_NAME", outcome_col="EVENT_CONCEPT_NAME",
        truth_col="GROUND_TRUTH")
    harpaz_matched.to_csv(REVISION_DIR / "revision_harpaz_validation.csv", index=False)

    result = evaluate_performance(harpaz_matched, "Harpaz 2014", out)
    summary_rows.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REVISION_DIR / "revision_external_validation_summary.csv",
                      index=False)

    # ── Export crosswalk and reference set expansion candidates ────────
    out.write("\n" + "=" * 70 + "\n")
    out.write("  Reference Set Expansion Candidates\n")
    out.write("=" * 70 + "\n")

    # Combine all matched pairs from external sets
    all_matched = []
    for name, df in [("OMOP", omop_matched), ("EU-ADR", euadr_matched),
                     ("Harpaz", harpaz_matched)]:
        m = df[df["matched"]].copy()
        m["source"] = name
        all_matched.append(m)

    if all_matched:
        combined = pd.concat(all_matched, ignore_index=True)

        # Load existing reference set to avoid duplicates
        existing = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
        existing_pairs = set(
            zip(existing["active_ingredient"].str.lower(),
                existing["reaction"].str.lower().str.lstrip("• "))
        )

        # Find new positive/negative controls not in existing set
        new_controls = []
        for _, r in combined.iterrows():
            if r["matched_drug"] is None:
                continue
            pair = (r["matched_drug"].lower(), r["matched_reaction"].lower())
            # Check not already in reference set (rough substring check)
            already = any(pair[0] in ep[0] and pair[1] in ep[1]
                         for ep in existing_pairs)
            if not already:
                new_controls.append({
                    "source": r["source"],
                    "ref_drug": r["ref_drug"],
                    "ref_outcome": r["ref_outcome"],
                    "ground_truth": r["ground_truth"],
                    "daen_drug": r["matched_drug"],
                    "daen_reaction": r["matched_reaction"],
                    "n_reports": r["n_reports"],
                    "ebgm": r["ebgm"],
                    "eb05": r["eb05"],
                    "signal_ebgm": r["signal_ebgm"],
                    "consensus": r["consensus"],
                })

        new_df = pd.DataFrame(new_controls)
        if len(new_df) > 0:
            new_pos = new_df[new_df["ground_truth"] == 1]
            new_neg = new_df[new_df["ground_truth"] == 0]
            out.write(f"\n  New matched pairs not in existing reference set:\n")
            out.write(f"    Positive controls: {len(new_pos)}\n")
            out.write(f"    Negative controls: {len(new_neg)}\n")

            new_df.to_csv(REVISION_DIR / "revision_reference_expansion_candidates.csv",
                          index=False)
            out.write(f"\n  Saved to: revision_reference_expansion_candidates.csv\n")
        else:
            out.write(f"\n  No new candidate pairs found.\n")

    # Save crosswalk for documentation
    crosswalk_rows = [{"us_name": k, "au_name": v} for k, v in DRUG_NAME_MAP.items()]
    pd.DataFrame(crosswalk_rows).to_csv(
        REFERENCE_DIR / "drug_name_crosswalk.csv", index=False)
    out.write(f"\n  Drug name crosswalk saved to: data/reference/drug_name_crosswalk.csv"
              f" ({len(DRUG_NAME_MAP)} mappings)\n")

    elapsed = time.time() - t0
    out.write(f"\n{'=' * 70}\n")
    out.write(f"  External validation complete ({elapsed:.0f}s)\n")
    out.write(f"{'=' * 70}\n")

    # Save report
    (REVISION_DIR / "revision_external_validation_report.txt").write_text(
        report.getvalue())


if __name__ == "__main__":
    main()
