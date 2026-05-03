"""
Healthcare Dataset Integration for Mem0
=========================================

Integrates the following datasets into Mem0's three-tier memory architecture:

SEMANTIC MEMORY (global, shared across all patients):
  • MedQuAD             – medical Q&A knowledge base
  • Medical Flashcards  – clinical fact cards
  • Synthea COVID-19 (100k_synthea_covid19_csv)
      – population-level condition & medication statistics

EPISODIC MEMORY (scoped per patient / user_id):
  All CSVs from synthea_sample_data_csv_nov2021/csv/:
    conditions.csv       allergies.csv      careplans.csv
    medications.csv      procedures.csv     encounters.csv
    immunizations.csv    observations.csv   imaging_studies.csv
    devices.csv          supplies.csv       payer_transitions.csv

SHORT-TERM MEMORY:
  SQLite – last K conversation messages per patient session.
"""

import csv
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthcareDatasetLoader:
    """Loads all healthcare datasets and prepares them for Mem0 integration."""

    def __init__(self, datasets_dir: str = "Datasets"):
        self.datasets_dir = datasets_dir

        # ── Semantic sources ────────────────────────────────────────────────
        self.medquad_path = os.path.join(datasets_dir, "medquad.csv", "medquad.csv")
        self.flashcards_path = os.path.join(
            datasets_dir, "medquad.csv", "medical_flashcards_semantic.csv"
        )

        # ── Standard Synthea (episodic) ─────────────────────────────────────
        self.synthea_dir = os.path.join(
            datasets_dir, "synthea_sample_data_csv_nov2021", "csv"
        )

        # ── COVID-19 Synthea 100k (semantic – population-level) ─────────────
        self.covid_synthea_dir = os.path.join(
            datasets_dir,
            "synthea_sample_data_csv_nov2021",
            "100k_synthea_covid19_csv",
        )

    # ===================================================================
    # SEMANTIC MEMORY LOADERS
    # ===================================================================

    def load_medquad(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Load MedQuAD QA pairs as semantic memories."""
        rows: List[Dict[str, str]] = []
        try:
            with open(self.medquad_path, "r", encoding="utf-8") as f:
                for idx, row in enumerate(csv.DictReader(f)):
                    if limit and idx >= limit:
                        break
                    rows.append(
                        {
                            "question": row.get("question", "").strip(),
                            "answer": row.get("answer", "").strip()[:500],
                            "source": row.get("source", "").strip(),
                            "focus_area": row.get("focus_area", "").strip(),
                        }
                    )
            logger.info(f"Loaded {len(rows)} MedQuAD records")
        except FileNotFoundError:
            logger.error(f"MedQuAD not found: {self.medquad_path}")
        return rows

    def load_medical_flashcards(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Load medical flashcard Q&A pairs as semantic memories."""
        rows: List[Dict[str, str]] = []
        try:
            with open(self.flashcards_path, "r", encoding="utf-8") as f:
                for idx, row in enumerate(csv.DictReader(f)):
                    if limit and idx >= limit:
                        break
                    rows.append(
                        {
                            "input": row.get("input", "").strip(),
                            "output": row.get("output", "").strip()[:500],
                            "instruction": row.get("instruction", "").strip(),
                        }
                    )
            logger.info(f"Loaded {len(rows)} medical flashcards")
        except FileNotFoundError:
            logger.error(f"Flashcards not found: {self.flashcards_path}")
        return rows

    def load_covid_conditions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load unique COVID-19 conditions from 100k Synthea dataset (SEMANTIC).
        De-duplicated by DESCRIPTION; sorted by patient count descending.
        """
        filepath = os.path.join(self.covid_synthea_dir, "conditions.csv")
        seen: Dict[str, int] = {}
        codes: Dict[str, str] = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    desc = row.get("DESCRIPTION", "").strip()
                    code = row.get("CODE", "").strip()
                    if desc:
                        seen[desc] = seen.get(desc, 0) + 1
                        codes.setdefault(desc, code)
            items = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            if limit:
                items = items[:limit]
            results = [{"description": d, "code": codes[d], "count": c} for d, c in items]
            logger.info(f"Loaded {len(results)} unique COVID-19 condition types")
        except FileNotFoundError:
            logger.error(f"COVID conditions not found: {filepath}")
            results = []
        return results

    def load_covid_medications(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load unique COVID-19 medications from 100k Synthea dataset (SEMANTIC).
        De-duplicated by DESCRIPTION; sorted by patient count descending.
        """
        filepath = os.path.join(self.covid_synthea_dir, "medications.csv")
        seen: Dict[str, int] = {}
        codes: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    desc = row.get("DESCRIPTION", "").strip()
                    code = row.get("CODE", "").strip()
                    reason = row.get("REASONDESCRIPTION", "").strip()
                    if desc:
                        seen[desc] = seen.get(desc, 0) + 1
                        codes.setdefault(desc, code)
                        if reason:
                            reasons.setdefault(desc, reason)
            items = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            if limit:
                items = items[:limit]
            results = [
                {"description": d, "code": codes.get(d, ""), "reason": reasons.get(d, ""), "count": c}
                for d, c in items
            ]
            logger.info(f"Loaded {len(results)} unique COVID-19 medication types")
        except FileNotFoundError:
            logger.error(f"COVID medications not found: {filepath}")
            results = []
        return results

    # ===================================================================
    # EPISODIC MEMORY LOADERS  (all scoped to PATIENT column)
    # ===================================================================

    def _csv_rows_for_patient(
        self,
        filename: str,
        patient_id: Optional[str],
        patient_col: str = "PATIENT",
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Generic helper: stream a CSV from self.synthea_dir and return rows.
        If patient_id is given, only rows where patient_col == patient_id are returned.
        """
        filepath = os.path.join(self.synthea_dir, filename)
        rows: List[Dict[str, str]] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if patient_id and row.get(patient_col, "").strip() != patient_id:
                        continue
                    rows.append({k: (v or "").strip() for k, v in row.items()})
                    if limit and len(rows) >= limit:
                        break
        except FileNotFoundError:
            logger.warning(f"File not found (skipping): {filepath}")
        return rows

    # ── Patients ─────────────────────────────────────────────────────────

    def load_synthea_patients(
        self, limit: Optional[int] = None, use_covid: bool = False
    ) -> List[Dict[str, Any]]:
        """Load ALL Synthea patient demographics (no limit by default)."""
        source_dir = self.covid_synthea_dir if use_covid else self.synthea_dir
        patients: List[Dict[str, Any]] = []
        filepath = os.path.join(source_dir, "patients.csv")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for idx, row in enumerate(csv.DictReader(f)):
                    if limit and idx >= limit:
                        break
                    patients.append(
                        {
                            "PATIENT_ID": row.get("Id", "").strip(),
                            "FIRST": row.get("FIRST", "").strip(),
                            "LAST": row.get("LAST", "").strip(),
                            "GENDER": row.get("GENDER", "").strip(),
                            "RACE": row.get("RACE", "").strip(),
                            "ETHNICITY": row.get("ETHNICITY", "").strip(),
                            "BIRTHDATE": row.get("BIRTHDATE", "").strip(),
                            "DEATHDATE": row.get("DEATHDATE", "").strip(),
                            "MARITAL": row.get("MARITAL", "").strip(),
                            "ADDRESS": row.get("ADDRESS", "").strip(),
                            "CITY": row.get("CITY", "").strip(),
                            "STATE": row.get("STATE", "").strip(),
                        }
                    )
            logger.info(f"Loaded {len(patients)} patients")
        except FileNotFoundError:
            logger.error(f"Patients file not found: {filepath}")
        return patients

    # ── Conditions ────────────────────────────────────────────────────────

    def load_synthea_conditions(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("conditions.csv", patient_id, limit=limit)

    # ── Medications ───────────────────────────────────────────────────────

    def load_synthea_medications(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("medications.csv", patient_id, limit=limit)

    # ── Encounters ────────────────────────────────────────────────────────

    def load_synthea_encounters(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("encounters.csv", patient_id, limit=limit)

    # ── Allergies ─────────────────────────────────────────────────────────

    def load_synthea_allergies(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("allergies.csv", patient_id, limit=limit)

    # ── Careplans ─────────────────────────────────────────────────────────

    def load_synthea_careplans(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("careplans.csv", patient_id, limit=limit)

    # ── Procedures ───────────────────────────────────────────────────────

    def load_synthea_procedures(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("procedures.csv", patient_id, limit=limit)

    # ── Immunizations ─────────────────────────────────────────────────────

    def load_synthea_immunizations(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("immunizations.csv", patient_id, limit=limit)

    # ── Observations ──────────────────────────────────────────────────────

    def load_synthea_observations(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Observations are extremely numerous. Per-patient limit defaults to 20
        to keep embedding volume manageable. Pass limit=None for unlimited.
        """
        return self._csv_rows_for_patient("observations.csv", patient_id, limit=limit)

    # ── Imaging Studies ───────────────────────────────────────────────────

    def load_synthea_imaging_studies(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("imaging_studies.csv", patient_id, limit=limit)

    # ── Devices ───────────────────────────────────────────────────────────

    def load_synthea_devices(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("devices.csv", patient_id, limit=limit)

    # ── Supplies ──────────────────────────────────────────────────────────

    def load_synthea_supplies(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient("supplies.csv", patient_id, limit=limit)

    # ── Payer Transitions ─────────────────────────────────────────────────

    def load_synthea_payer_transitions(
        self, patient_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._csv_rows_for_patient(
            "payer_transitions.csv", patient_id, patient_col="PATIENT", limit=limit
        )

    # ===================================================================
    # FORMATTERS  (Semantic)
    # ===================================================================

    def format_semantic_memory(self, qa: Dict[str, str]) -> str:
        return f"Medical Knowledge: {qa['question']} Answer: {qa['answer']}"

    def format_flashcard(self, card: Dict[str, str]) -> str:
        return f"Medical Fact: {card['input']} Answer: {card['output']}"

    def format_covid_condition(self, item: Dict[str, Any]) -> str:
        parts = [f"COVID-19 clinical condition: {item['description']}"]
        if item.get("code"):
            parts.append(f"(SNOMED: {item['code']})")
        if item.get("count"):
            parts.append(f"— observed in {item['count']} patients in the 100k COVID-19 population dataset.")
        return " ".join(parts)

    def format_covid_medication(self, item: Dict[str, Any]) -> str:
        parts = [f"COVID-19 related medication: {item['description']}"]
        if item.get("reason"):
            parts.append(f"prescribed for {item['reason']}")
        if item.get("count"):
            parts.append(f"— used by {item['count']} patients in the 100k COVID-19 population dataset.")
        return " ".join(parts)

    # ===================================================================
    # FORMATTERS  (Episodic)
    # ===================================================================

    def format_patient_demographics(self, p: Dict[str, Any]) -> str:
        death = f", deceased {p['DEATHDATE']}" if p.get("DEATHDATE") else ""
        marital = f", marital status: {p['MARITAL']}" if p.get("MARITAL") else ""
        return (
            f"Patient {p['FIRST']} {p['LAST']}, born {p['BIRTHDATE']}{death}, "
            f"{p['GENDER']}, {p['RACE']} {p['ETHNICITY']}{marital}, "
            f"lives in {p['CITY']}, {p['STATE']}."
        )

    def format_episodic_condition(self, row: Dict[str, Any]) -> str:
        stop = f" until {row['STOP']}" if row.get("STOP") else " (ongoing)"
        return f"Patient diagnosed with {row['DESCRIPTION']} starting {row['START']}{stop}."

    def format_episodic_medication(self, row: Dict[str, Any]) -> str:
        stop = f" until {row['STOP']}" if row.get("STOP") else " (ongoing)"
        reason = f" for {row['REASONDESCRIPTION']}" if row.get("REASONDESCRIPTION") else ""
        return f"Patient prescribed {row['DESCRIPTION']} starting {row['START']}{stop}{reason}."

    def format_episodic_encounter(self, row: Dict[str, Any]) -> str:
        return (
            f"Patient had a {row.get('ENCOUNTERCLASS', 'clinical')} encounter "
            f"({row.get('DESCRIPTION', '')}) from {row.get('START', '')} to {row.get('STOP', '')}."
        )

    def format_episodic_allergy(self, row: Dict[str, Any]) -> str:
        stop = f" until {row['STOP']}" if row.get("STOP") else " (ongoing)"
        severity = f" — severity: {row['SEVERITY1']}" if row.get("SEVERITY1") else ""
        reaction = f" (reaction: {row['DESCRIPTION1']})" if row.get("DESCRIPTION1") else ""
        return (
            f"Patient has allergy to {row.get('DESCRIPTION', '')} (category: {row.get('CATEGORY', '')}, "
            f"type: {row.get('TYPE', '')}){reaction}{severity}, noted {row.get('START', '')}{stop}."
        )

    def format_episodic_careplan(self, row: Dict[str, Any]) -> str:
        stop = f" until {row['STOP']}" if row.get("STOP") else " (ongoing)"
        reason = f" for {row['REASONDESCRIPTION']}" if row.get("REASONDESCRIPTION") else ""
        return f"Patient on care plan: {row.get('DESCRIPTION', '')} starting {row.get('START', '')}{stop}{reason}."

    def format_episodic_procedure(self, row: Dict[str, Any]) -> str:
        reason = f" for {row['REASONDESCRIPTION']}" if row.get("REASONDESCRIPTION") else ""
        return f"Patient underwent procedure: {row.get('DESCRIPTION', '')} on {row.get('START', '')}{reason}."

    def format_episodic_immunization(self, row: Dict[str, Any]) -> str:
        return f"Patient received immunization: {row.get('DESCRIPTION', '')} on {row.get('DATE', '')}."

    def format_episodic_observation(self, row: Dict[str, Any]) -> str:
        value = row.get("VALUE", "")
        units = row.get("UNITS", "")
        val_str = f": {value} {units}".strip() if value else ""
        return (
            f"Patient observation — {row.get('DESCRIPTION', '')} ({row.get('CATEGORY', '')})"
            f"{val_str} on {row.get('DATE', '')}."
        )

    def format_episodic_imaging_study(self, row: Dict[str, Any]) -> str:
        return (
            f"Patient had imaging study: {row.get('MODALITY_DESCRIPTION', '')} "
            f"of {row.get('BODYSITE_DESCRIPTION', '')} on {row.get('DATE', '')}."
        )

    def format_episodic_device(self, row: Dict[str, Any]) -> str:
        stop = f" until {row['STOP']}" if row.get("STOP") else " (still implanted)"
        return f"Patient has device: {row.get('DESCRIPTION', '')} implanted {row.get('START', '')}{stop}."

    def format_episodic_supply(self, row: Dict[str, Any]) -> str:
        qty = f", quantity: {row['QUANTITY']}" if row.get("QUANTITY") else ""
        return f"Patient received supply: {row.get('DESCRIPTION', '')}{qty} on {row.get('DATE', '')}."

    def format_episodic_payer_transition(self, row: Dict[str, Any]) -> str:
        payer_name = row.get("OWNERNAME", row.get("PAYER", "unknown"))
        return (
            f"Patient insurance: {payer_name} (ownership: {row.get('OWNERSHIP', '')}) "
            f"from {row.get('START_YEAR', '')} to {row.get('END_YEAR', '')}."
        )
