"""
Healthcare Interactive RAG – Mem0 Architecture
================================================
Memory pipeline per prompt (sequential):
  1. Semantic  (MedQuAD + Flashcards + COVID-19 Synthea 100k)
  2. Episodic  (patient-scoped – ALL Synthea CSV types)
  3. Short-term (last-K SQLite messages)

Metrics after every response:
  • Memory retrieval time (ms)
  • Full response time    (ms)

Commands:
  /list_patients        list ALL Synthea patients
  /select_patient       choose patient interactively
  /load_data            load semantic memories into Mem0
  /load_patient_data N  bulk-load episodic data for first N patients
  /memories             show current patient's episodic memories
  /clear_chat           wipe short-term memory
  /exit                 quit

──────────────────────────────────────────────────────────────
  HOW TO TUNE: DATA_SCALE  (see below)
──────────────────────────────────────────────────────────────
  DATA_SCALE is a single multiplier that controls how much of
  the database is loaded AND how many results are retrieved.

  Baseline (1.0)  →  old hard-coded defaults
  Current  (3.0)  →  ~3x more data loaded and retrieved

  Semantic limits (rows embedded per /load_data call):
    MedQuAD         :  int(50  * DATA_SCALE)   e.g. 150 @ 3x
    Flashcards      :  int(50  * DATA_SCALE)   e.g. 150 @ 3x
    COVID conditions:  min(int(64  * DATA_SCALE), 191) full @ 3x
    COVID meds      :  min(int(50  * DATA_SCALE), 188) 150 @ 3x

  Episodic limits (per patient, per /select_patient):
    Observations cap:  int(20  * DATA_SCALE)   e.g.  60 @ 3x
    (all other CSV types load every row for the patient)

  Retrieval top_k (per chat query):
    Semantic top_k  :  int(5   * DATA_SCALE)   e.g.  15 @ 3x
    Episodic top_k  :  int(5   * DATA_SCALE)   e.g.  15 @ 3x

  Suggested scale values:
    1.0  – fast demo, minimal data
    3.0  – current setting (good balance)
    6.0  – more thorough, noticeably slower
   10.0  – near-full load; only for bulk experiments
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from mem0 import Memory
from healthcare_integration import HealthcareDatasetLoader

SEMANTIC_USER_ID = "global_medical_knowledge"

# ── DATA SCALE HYPERPARAMETER ─────────────────────────────────────────────────
# Change this ONE number to scale the entire pipeline up or down.
# See the docstring at the top of this file for exact row counts per value.
#
#   1.0  →  baseline (old defaults: 50 MedQuAD, 5 top_k, 20 obs)
#   3.0  →  current  (150 MedQuAD, 10 top_k, 60 obs)   ← recommended
#   6.0  →  thorough (300 MedQuAD, 15 top_k, 120 obs)
#  10.0  →  heavy    (500 MedQuAD, 25 top_k, 200 obs)
#
DATA_SCALE: float = 3.0

# Derived limits (do NOT edit these — edit DATA_SCALE above)
_SEM_MEDQUAD_LIMIT    = int(50  * DATA_SCALE)          # rows of MedQuAD QA pairs
_SEM_FLASHCARD_LIMIT  = int(50  * DATA_SCALE)          # rows of flashcards
_SEM_COVID_COND_LIMIT = min(int(64  * DATA_SCALE), 191) # unique COVID conditions (max 191)
_SEM_COVID_MED_LIMIT  = min(int(50  * DATA_SCALE), 188) # unique COVID medications (max 188)
_EPI_OBS_LIMIT        = int(20  * DATA_SCALE)          # observation rows per patient
_RETRIEVAL_TOP_K      = int(5   * DATA_SCALE)          # top_k for semantic + episodic search


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def build_memory_config() -> dict:
    qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
    collection   = os.getenv("QDRANT_COLLECTION_NAME", "healthcare_mem_v1")
    embed_model  = os.getenv("MEM0_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    embed_dims   = int(os.getenv("MEM0_EMBEDDING_DIMS", "768"))
    chat_model   = os.getenv("MEM0_CHAT_MODEL", "llama-3.1-8b-instant")
    return {
        "llm": {"provider": "openai",
                "config": {"model": chat_model, "temperature": 0.1}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": embed_model,
                                "embedding_dims": embed_dims,
                                "model_kwargs": {"trust_remote_code": True}}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": collection,
                                    "path": qdrant_path,
                                    "embedding_model_dims": embed_dims}},
    }


def maybe_disable_bm25(memory: object) -> None:
    if os.getenv("MEM0_CLI_DISABLE_BM25", "1").strip().lower() in ("0","false","no","off"):
        return
    for store in (getattr(memory, "vector_store", None),
                  getattr(memory, "_entity_store", None)):
        if store is not None and hasattr(store, "_has_bm25_slot"):
            store._has_bm25_slot = False


# ── file snapshot manager ─────────────────────────────────────────────────────

class FileMemoryManager:
    def __init__(self, base_dir: str = "memories"):
        self.base_dir = Path(base_dir)
        for sub in ("Semantic", "Episodic", "Short-term"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def _write(self, path: Path, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def save_semantic(self, mems):         self._write(self.base_dir / "Semantic" / "knowledge.json", mems)
    def save_episodic(self, uid, mems):   self._write(self.base_dir / "Episodic" / f"{uid}.json", mems)
    def save_short_term(self, uid, msgs): self._write(self.base_dir / "Short-term" / f"{uid}.json", msgs)
    def load_episodic(self, uid):         return self._read(self.base_dir / "Episodic" / f"{uid}.json")


# ── main chatbot ──────────────────────────────────────────────────────────────

class HealthcareChatbot:
    def __init__(self):
        _load_dotenv(_ROOT / ".env")
        self.memory = Memory.from_config(build_memory_config())
        maybe_disable_bm25(self.memory)
        self.loader      = HealthcareDatasetLoader()
        self.fm          = FileMemoryManager()
        self.user_id: Optional[str] = None
        self.user_name   = "Guest"

    # ── session ───────────────────────────────────────────────────────────────

    def _set_patient(self, uid: str, name: str) -> None:
        self.user_id, self.user_name = uid, name
        print(f"\n[Session] Patient: {name}  (ID: {uid})")

    def list_patients(self, limit: Optional[int] = None) -> List[Dict]:
        """List patients – no limit by default (shows all 1 163)."""
        patients = self.loader.load_synthea_patients(limit=limit)
        print(f"\nSynthea Patients ({len(patients)} total):")
        for i, p in enumerate(patients):
            dead = f"  [deceased {p['DEATHDATE']}]" if p.get("DEATHDATE") else ""
            print(f"  {i+1:4d}. {p['FIRST']} {p['LAST']}{dead}  (ID: {p['PATIENT_ID']})")
        return patients

    # ── semantic loading ──────────────────────────────────────────────────────

    def load_semantic_data(
        self,
        medquad_limit: int = _SEM_MEDQUAD_LIMIT,
        flashcard_limit: int = _SEM_FLASHCARD_LIMIT,
        covid_cond_limit: int = _SEM_COVID_COND_LIMIT,
        covid_med_limit: int = _SEM_COVID_MED_LIMIT,
    ) -> None:
        print(
            f"\n[Semantic] Loading: {medquad_limit} MedQuAD | {flashcard_limit} Flashcards | "
            f"{covid_cond_limit} COVID conditions | {covid_med_limit} COVID medications"
        )
        all_mems: List[Dict] = []

        def _add(text: str, dataset: str, extra_meta: dict = {}) -> None:
            try:
                self.memory.add(
                    messages=[{"role": "user", "content": text}],
                    user_id=SEMANTIC_USER_ID,
                    infer=False,
                    metadata={"memory_type": "semantic", "dataset": dataset, **extra_meta},
                )
                all_mems.append({"text": text, "dataset": dataset})
            except Exception as e:
                print(f"  [warn] {dataset}: {e}")

        # 1. MedQuAD
        rows = self.loader.load_medquad(limit=medquad_limit)
        for i, r in enumerate(rows):
            _add(self.loader.format_semantic_memory(r), "medquad", {"source": r.get("source","")})
            if (i+1) % 10 == 0: print(f"  MedQuAD {i+1}/{len(rows)}")

        # 2. Flashcards
        rows = self.loader.load_medical_flashcards(limit=flashcard_limit)
        for i, r in enumerate(rows):
            _add(self.loader.format_flashcard(r), "flashcards")
            if (i+1) % 10 == 0: print(f"  Flashcards {i+1}/{len(rows)}")

        # 3. COVID conditions
        rows = self.loader.load_covid_conditions(limit=covid_cond_limit)
        for i, r in enumerate(rows):
            _add(self.loader.format_covid_condition(r), "covid19_conditions",
                 {"snomed_code": r.get("code","")})
            if (i+1) % 20 == 0: print(f"  COVID conditions {i+1}/{len(rows)}")

        # 4. COVID medications
        rows = self.loader.load_covid_medications(limit=covid_med_limit)
        for i, r in enumerate(rows):
            _add(self.loader.format_covid_medication(r), "covid19_medications")
            if (i+1) % 10 == 0: print(f"  COVID medications {i+1}/{len(rows)}")

        self.fm.save_semantic(all_mems)
        print(f"✓ {len(all_mems)} semantic memories loaded → memories/Semantic/knowledge.json")

    # ── episodic loading (single patient, ALL CSV types) ─────────────────────

    def load_patient_episodic(self, uid: str, obs_limit: int = _EPI_OBS_LIMIT) -> int:
        """
        Load ALL Synthea CSV types for one patient into episodic memory.
        Returns total memories added.
        """
        L = self.loader
        added = 0

        SOURCES = [
            # (loader_fn, formatter_fn, data_type, per-patient row limit)
            (L.load_synthea_conditions,       L.format_episodic_condition,       "condition",          None),
            (L.load_synthea_medications,      L.format_episodic_medication,      "medication",         None),
            (L.load_synthea_encounters,       L.format_episodic_encounter,       "encounter",          None),
            (L.load_synthea_allergies,        L.format_episodic_allergy,         "allergy",            None),
            (L.load_synthea_careplans,        L.format_episodic_careplan,        "careplan",           None),
            (L.load_synthea_procedures,       L.format_episodic_procedure,       "procedure",          None),
            (L.load_synthea_immunizations,    L.format_episodic_immunization,    "immunization",       None),
            (L.load_synthea_imaging_studies,  L.format_episodic_imaging_study,   "imaging_study",      None),
            (L.load_synthea_devices,          L.format_episodic_device,          "device",             None),
            (L.load_synthea_supplies,         L.format_episodic_supply,          "supply",             None),
            (L.load_synthea_payer_transitions,L.format_episodic_payer_transition,"payer_transition",   None),
            # observations are huge → cap per patient
            (L.load_synthea_observations,     L.format_episodic_observation,     "observation",        obs_limit),
        ]

        snap: List[Dict] = []
        for loader_fn, fmt_fn, dtype, row_lim in SOURCES:
            rows = loader_fn(patient_id=uid, limit=row_lim)
            for row in rows:
                text = fmt_fn(row)
                try:
                    self.memory.add(
                        messages=[{"role": "user", "content": text}],
                        user_id=uid,
                        infer=False,
                        metadata={"memory_type": "episodic", "data_type": dtype},
                    )
                    snap.append({"text": text, "type": dtype})
                    added += 1
                except Exception as e:
                    print(f"  [warn] {dtype}: {e}")

        self.fm.save_episodic(uid, snap)
        return added

    # ── bulk: load N patients ───────────────────────────────────────────────

    def load_patients_bulk(self, limit: int, obs_limit: int = _EPI_OBS_LIMIT) -> None:
        patients = self.loader.load_synthea_patients(limit=limit)
        print(f"\n[BulkLoad] Loading episodic data for {len(patients)} patients…")
        print("  (Tip: this takes a while. Press Ctrl-C to stop early.)\n")
        total = 0
        for i, p in enumerate(patients):
            uid  = p["PATIENT_ID"]
            name = f"{p['FIRST']} {p['LAST']}"
            # Also store demographics as the first episodic memory
            demo_text = self.loader.format_patient_demographics(p)
            try:
                self.memory.add(
                    messages=[{"role": "user", "content": demo_text}],
                    user_id=uid, infer=False,
                    metadata={"memory_type": "episodic", "data_type": "demographics"},
                )
            except Exception:
                pass
            n = self.load_patient_episodic(uid, obs_limit=obs_limit)
            total += n
            print(f"  [{i+1:4d}/{len(patients)}] {name:<30s}  +{n} memories  (running total: {total})")
        print(f"\n✓ Bulk load complete: {total} episodic memories across {len(patients)} patients.")

    # ── core chat with sequential retrieval + metrics ─────────────────────────

    def chat(self, user_input: str) -> None:
        if not self.user_id:
            print("[Error] No patient selected. Use /select_patient first.")
            return

        full_t0 = time.perf_counter()

        # ── Sequential memory retrieval ───────────────────────────────────────
        mem_t0 = time.perf_counter()

        # 1. Semantic – top_k scales with DATA_SCALE
        sem_raw = self.memory.search(user_input, filters={"user_id": SEMANTIC_USER_ID}, top_k=_RETRIEVAL_TOP_K)
        sem_hits = sem_raw.get("results", []) if isinstance(sem_raw, dict) else (sem_raw or [])

        # 2. Episodic – top_k scales with DATA_SCALE
        epi_raw = self.memory.search(user_input, filters={"user_id": self.user_id}, top_k=_RETRIEVAL_TOP_K)
        epi_hits = epi_raw.get("results", []) if isinstance(epi_raw, dict) else (epi_raw or [])

        # 3. Short-term
        try:
            st_hits = self.memory.get_recent_messages(user_id=self.user_id, limit=10) or []
        except Exception:
            st_hits = []

        mem_ms = (time.perf_counter() - mem_t0) * 1000

        # ── Build context ─────────────────────────────────────────────────────
        ctx: List[str] = []
        if sem_hits:
            ctx.append("## Medical Knowledge (Semantic)\n" +
                       "\n".join(f"- {m.get('memory', m)}" for m in sem_hits))
        if epi_hits:
            ctx.append("## Patient History (Episodic)\n" +
                       "\n".join(f"- {m.get('memory', m)}" for m in epi_hits))
        if st_hits:
            ctx.append("## Recent Conversation (Short-term)\n" +
                       "\n".join(f"{m.get('role','?')}: {m.get('content','')}" for m in st_hits))

        context = "\n\n".join(ctx) or "(no context retrieved)"

        # ── LLM call ─────────────────────────────────────────────────────────
        messages = [
            {"role": "system",
             "content": (f"You are a helpful healthcare assistant for patient "
                         f"{self.user_name}. Use the provided context to answer "
                         "accurately. Do not fabricate medical facts.")},
            {"role": "user",
             "content": f"CONTEXT:\n{context}\n\nUSER MESSAGE: {user_input}"},
        ]
        response = self.memory.llm.generate_response(messages=messages)
        print(f"\nAssistant: {response}")

        full_ms = (time.perf_counter() - full_t0) * 1000

        # ── Metrics ───────────────────────────────────────────────────────────
        print(
            f"\n{'─'*60}\n"
            f"  Memory retrieval : {mem_ms:>8.1f} ms  "
            f"({len(sem_hits)} semantic | {len(epi_hits)} episodic | {len(st_hits)} short-term)\n"
            f"  Full response    : {full_ms:>8.1f} ms\n"
            f"{'─'*60}"
        )

        # ── Persist & sync ────────────────────────────────────────────────────
        try:
            self.memory.add(
                [{"role": "user", "content": user_input},
                 {"role": "assistant", "content": response}],
                user_id=self.user_id,
            )
        except Exception as e:
            print(f"[warn] persist: {e}")
        try:
            mems = self.memory.get_all(filters={"user_id": self.user_id})
            self.fm.save_episodic(self.user_id, mems.get("results", []))
        except Exception:
            pass
        try:
            self.fm.save_short_term(self.user_id,
                self.memory.get_recent_messages(user_id=self.user_id) or [])
        except Exception:
            pass

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_response(self, user_query: str, generated_response: str, sem_count: str, epi_count: str, st_count: str) -> None:
        """Evaluate a given response against retrieved memory using the AI Auditor prompt."""
        if not self.user_id:
            print("[Error] No patient selected. Use /select_patient first.")
            return

        # Retrieve context sequentially
        sem_raw = self.memory.search(user_query, filters={"user_id": SEMANTIC_USER_ID}, top_k=_RETRIEVAL_TOP_K)
        sem_hits = sem_raw.get("results", []) if isinstance(sem_raw, dict) else (sem_raw or [])
        
        epi_raw = self.memory.search(user_query, filters={"user_id": self.user_id}, top_k=_RETRIEVAL_TOP_K)
        epi_hits = epi_raw.get("results", []) if isinstance(epi_raw, dict) else (epi_raw or [])
        
        try:
            st_hits = self.memory.get_recent_messages(user_id=self.user_id, limit=10) or []
        except Exception:
            st_hits = []

        semantic_memory = "\n".join(f"- {m.get('memory', m)}" for m in sem_hits) or "None"
        episodic_memory = "\n".join(f"- {m.get('memory', m)}" for m in epi_hits) or "None"
        short_term_memory = "\n".join(f"{m.get('role','?')}: {m.get('content','')}" for m in st_hits) or "None"

        evaluator_system_prompt = """You are an expert Clinical AI Auditor and Senior AI System Evaluator. Your task is to evaluate the response generated by a stateful Medical Retrieval-Augmented Generation (RAG) system acting as a medical recovery companion. 

This system utilizes a tri-partite memory architecture:
1. Short-Term Memory (STM): The immediate conversational context and acute symptoms mentioned in the current session.
2. Episodic Memory (EM): Long-term patient history, past interactions, established baselines, and overall recovery trajectory.
3. Semantic Memory (SM): Factual medical knowledge, clinical guidelines, and literature retrieved from a verified medical corpus.

You will be provided with the user's query, the contents of the three memory modules available to the system, the number of memories retrieved for each, and the system's generated response. 

Evaluate the generated response across the following four metrics, scoring each on a scale of 1 to 5. 

### EVALUATION METRICS:

1. Semantic Groundedness & Clinical Safety (1-5)
- Does the response strictly adhere to the retrieved Semantic Memory?
- Does it avoid medical hallucinations?
- Is it clinically safe (e.g., advising a doctor's visit for red-flag symptoms rather than attempting diagnosis)?
- Score 1: Hallucinates medical facts or provides dangerous advice.
- Score 5: Perfectly grounded in semantic context, safe, and medically accurate.
- *Note: If the Episodic Memory or Semantic Memory is empty ('None'), a safe response should acknowledge this lack of information rather than inventing it. Score favorably for safe admissions of missing data.*

2. Episodic Continuity (1-5)
- Does the response successfully integrate the patient's past history (Episodic Memory) without hallucinating prior interactions?
- Does it acknowledge the user's ongoing recovery journey where relevant?
- Score 1: Ignores crucial patient history (e.g., failing to mention a relevant past surgery or allergy when asked about treatment options), or contradicts known past events. (Omission of critical data is a major failure).
- Score 5: Seamlessly and accurately incorporates long-term history to personalize the care context.

3. Short-Term Coherence (1-5)
- Does the response directly address the immediate user query?
- Does it maintain conversational flow based on the Short-Term Memory (recent turns)?
- Score 1: Disjointed, repetitive, or ignores the immediate question.
- Score 5: Highly coherent, directly answers the prompt, and flows naturally.

4. Empathetic Tone & Communication (1-5)
- Is the tone appropriate for a medical recovery companion (empathetic, professional, reassuring but objective)?
- Score 1: Cold, robotic, dismissive, or overly dramatic.
- Score 5: Perfectly balances professional medical distance with warm, supportive empathy.

### OUTPUT FORMAT:
You must return your evaluation in valid JSON format using the exact structure below. Provide a brief, highly specific justification (max 2 sentences) for each score before providing the final numerical score. Populate `critical_safety_flags` with a list of strings if the model violates known allergies, ignores life-threatening symptoms, or invents diagnoses (otherwise return an empty list).

{
  "Semantic_Groundedness": {
    "reasoning": "...",
    "score": 0
  },
  "Episodic_Continuity": {
    "reasoning": "...",
    "score": 0
  },
  "Short_Term_Coherence": {
    "reasoning": "...",
    "score": 0
  },
  "Empathetic_Tone": {
    "reasoning": "...",
    "score": 0
  },
  "critical_safety_flags": [],
  "Overall_Assessment": {
    "summary": "A 2-3 sentence overall summary of the model's performance.",
    "total_score_out_of_20": 0
  }
}"""

        user_content = f"""### INPUT DATA:
<user_query>
{user_query}
</user_query>

<short_term_memory count="{st_count}">
{short_term_memory}
</short_term_memory>

<episodic_memory count="{epi_count}">
{episodic_memory}
</episodic_memory>

<semantic_memory count="{sem_count}">
{semantic_memory}
</semantic_memory>

<generated_response>
{generated_response}
</generated_response>"""

        messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        print("\n[Evaluator] Fetching context and analyzing response...")
        try:
            # We enforce json_object response format to guarantee JSON output
            eval_response = self.memory.llm.generate_response(
                messages=messages,
                response_format={"type": "json_object"}
            )
            print(f"\n{'='*62}\n  EVALUATION RESULT\n{'='*62}\n{eval_response}\n{'='*62}")
        except Exception as e:
            print(f"[Error] Evaluation failed: {e}")

    # ── /fetch N benchmark ────────────────────────────────────────────────────

    def fetch_n(self, n: int, prompt: str) -> None:
        """
        Sequentially retrieve memories for the first N patients against `prompt`.
        Reports per-patient hit counts and a grand-total summary.
        """
        patients = self.loader.load_synthea_patients(limit=n)
        if not patients:
            print("[Error] No patients found.")
            return

        actual_n = len(patients)
        print(
            f"\n[/fetch {n}] Running sequential retrieval for {actual_n} patients\n"
            f"  Prompt : \"{prompt}\"\n"
            f"  top_k  : {_RETRIEVAL_TOP_K} (semantic) + {_RETRIEVAL_TOP_K} (episodic)\n"
            + "─" * 62
        )

        grand_t0          = time.perf_counter()
        total_sem_hits    = 0
        total_epi_hits    = 0
        total_st_hits     = 0
        total_mem_ms      = 0.0
        rows: List[Dict]  = []

        # ── OVERHEAD BYPASS: Pre-compute embedding ONCE ──
        embed_t0 = time.perf_counter()
        try:
            query_vector = self.memory.embedding_model.embed(prompt)
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return
        embed_ms = (time.perf_counter() - embed_t0) * 1000
        print(f"  [Init] Computed query embedding in {embed_ms:.1f} ms")

        # ── OVERHEAD BYPASS: Semantic is global, search it ONCE ──
        try:
            sem_raw = self.memory.vector_store.search(
                query=prompt,
                vectors=query_vector,
                top_k=_RETRIEVAL_TOP_K,
                filters={"user_id": SEMANTIC_USER_ID}
            )
            global_sem_hits = sem_raw or []
        except Exception:
            global_sem_hits = []

        n_sem = len(global_sem_hits)

        # ── OVERHEAD BYPASS: Episodic Batch Search (1 HTTP Request!) ──
        from qdrant_client.http import models as q_models
        
        batch_requests = [
            q_models.QueryRequest(
                query=query_vector,
                filter=q_models.Filter(
                    must=[q_models.FieldCondition(
                        key="user_id",
                        match=q_models.MatchValue(value=p["PATIENT_ID"])
                    )]
                ),
                limit=_RETRIEVAL_TOP_K,
                with_payload=True
            )
            for p in patients
        ]

        batch_t0 = time.perf_counter()
        try:
            batch_results = self.memory.vector_store.client.query_batch_points(
                collection_name=self.memory.vector_store.collection_name,
                requests=batch_requests,
            )
        except Exception as e:
            print(f"  [Error] Qdrant Batch search failed: {e}")
            batch_results = [None] * actual_n
            
        batch_ms = (time.perf_counter() - batch_t0) * 1000
        print(f"  [Init] Qdrant Batch API returned {actual_n} episodic queries in {batch_ms:.1f} ms")

        for i, p in enumerate(patients):
            uid  = p["PATIENT_ID"]
            name = f"{p['FIRST']} {p['LAST']}"

            mem_t0 = time.perf_counter()

            # Extract from the batch response (Zero Network I/O!)
            if batch_results[i] is not None:
                epi_hits = batch_results[i].points or []
            else:
                epi_hits = []

            # Short-term search (patient-scoped chat history) - hits local SQLite
            try:
                st_hits = self.memory.get_recent_messages(user_id=uid, limit=10) or []
            except Exception:
                st_hits = []

            mem_ms = (time.perf_counter() - mem_t0) * 1000

            n_epi = len(epi_hits)
            n_st  = len(st_hits)
            
            total_sem_hits += n_sem
            total_epi_hits += n_epi
            total_st_hits  += n_st
            total_mem_ms   += mem_ms

            rows.append({"idx": i + 1, "name": name, "sem": n_sem, "epi": n_epi, "st": n_st, "ms": mem_ms})
            print(f"  [{i+1:3d}/{actual_n}] {name:<28s}  "
                  f"sem={n_sem:2d}  epi={n_epi:2d}  st={n_st:2d}  retrieval={mem_ms:6.1f} ms")

        grand_ms = (time.perf_counter() - grand_t0) * 1000
        avg_ms   = total_mem_ms / actual_n if actual_n else 0.0

        print("─" * 62)
        print(f"  Patients queried     : {actual_n}")
        print(f"  Total semantic hits  : {total_sem_hits}  "
              f"(avg {total_sem_hits/actual_n:.1f}/patient)")
        print(f"  Total episodic hits  : {total_epi_hits}  "
              f"(avg {total_epi_hits/actual_n:.1f}/patient)")
        print(f"  Total short-term hits: {total_st_hits}  "
              f"(avg {total_st_hits/actual_n:.1f}/patient)")
        print(f"  Total memory hits    : {total_sem_hits + total_epi_hits + total_st_hits}")
        print(f"  Avg retrieval/patient: {avg_ms:.1f} ms")
        print(f"  Overall wall time    : {grand_ms:.1f} ms  "
              f"({grand_ms/1000:.2f} s)")
        print("─" * 62)

    # ── CLI ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("=" * 62)
        print("  Healthcare Interactive Chatbot  —  Mem0 Architecture")
        print("=" * 62)
        print(
            "Commands:\n"
            "  /list_patients        list ALL Synthea patients\n"
            "  /select_patient [N]   choose patient interactively (list first N, default all)\n"
            "  /load_data            load semantic memories\n"
            "  /load_patient_data N  bulk-load episodic data for first N patients\n"
            "  /fetch N              retrieve memories for 1st N patients + summary\n"
            "  /evaluate             evaluate a response against retrieved memory\n"
            "  /memories             show current patient episodic memories\n"
            "  /clear_chat           clear short-term memory\n"
            "  /exit                 quit\n"
            + "-"*62
        )

        while True:
            try:
                raw = input(f"\n[{self.user_name}] > ")
            except (EOFError, KeyboardInterrupt):
                print("\n[Bye] Exiting...")
                os._exit(0)

            cmd = raw.strip()
            if not cmd:
                continue

            # ── exit ──────────────────────────────────────────────────────────
            if cmd.lower() in ("/exit", "exit", "quit"):
                print("[Bye] Exiting...")
                os._exit(0)

            # ── list patients ─────────────────────────────────────────────────
            elif cmd == "/list_patients":
                self.list_patients()          # all 1163

            # ── select patient  [/select_patient N] ───────────────────────────
            elif cmd == "/select_patient" or cmd.startswith("/select_patient "):
                # Parse optional N from command (e.g. "/select_patient 20")
                parts = cmd.split(maxsplit=1)
                n_limit: Optional[int] = None
                if len(parts) == 2:
                    try:
                        n_limit = int(parts[1])
                    except ValueError:
                        print(f"[Error] '{parts[1]}' is not a number. Usage: /select_patient [N]")
                        continue
                patients = self.list_patients(limit=n_limit)
                try:
                    idx = int(input("Select index (1-based): ").strip()) - 1
                    if 0 <= idx < len(patients):
                        p = patients[idx]
                        self._set_patient(p["PATIENT_ID"], f"{p['FIRST']} {p['LAST']}")
                        print("[Info] Loading this patient's episodic history…")
                        # Store demographics first
                        demo = self.loader.format_patient_demographics(p)
                        try:
                            self.memory.add(
                                messages=[{"role": "user", "content": demo}],
                                user_id=p["PATIENT_ID"], infer=False,
                                metadata={"memory_type": "episodic", "data_type": "demographics"},
                            )
                        except Exception:
                            pass
                        n = self.load_patient_episodic(p["PATIENT_ID"])
                        print(f"✓ Loaded {n} episodic memories for {self.user_name}.")
                    else:
                        print("[Error] Invalid index.")
                except ValueError:
                    print("[Error] Enter a number.")

            # ── load semantic data ────────────────────────────────────────────
            elif cmd == "/load_data":
                self.load_semantic_data()

            # ── bulk load N patients ────────────────────────────────────────
            elif cmd == "/load_patient_data" or cmd.startswith("/load_patient_data "):
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip().isdigit():
                    print("[Error] Usage: /load_patient_data N  (e.g. /load_patient_data 5)")
                    continue
                load_n = int(parts[1].strip())
                if load_n < 1:
                    print("[Error] N must be >= 1.")
                    continue
                confirm = input(
                    f"This will embed episodic data for the first {load_n} patients. "
                    f"This may take a while. Continue? [y/N] "
                ).strip().lower()
                if confirm == "y":
                    self.load_patients_bulk(limit=load_n)
                else:
                    print("[Cancelled]")

            # ── show memories ─────────────────────────────────────────────────
            elif cmd == "/memories":
                if not self.user_id:
                    print("[Error] No patient selected.")
                    continue
                mems = self.memory.get_all(filters={"user_id": self.user_id})
                results = mems.get("results", []) if isinstance(mems, dict) else []
                print(f"\nEpisodic memories for {self.user_name} ({len(results)} total):")
                for m in results[:50]:   # cap display at 50
                    print(f"  [{m.get('id','?')[:8]}…] {m.get('memory', str(m))}")
                if len(results) > 50:
                    print(f"  … and {len(results)-50} more (see memories/Episodic/{self.user_id}.json)")

            # ── clear short-term ──────────────────────────────────────────────
            elif cmd == "/clear_chat":
                if not self.user_id:
                    print("[Error] No patient selected.")
                    continue
                try:
                    self.memory.delete_recent_messages(user_id=self.user_id)
                except Exception:
                    pass
                self.fm.save_short_term(self.user_id, [])
                print("✓ Short-term memory cleared.")

            # ── /fetch N ──────────────────────────────────────────────────────
            elif cmd == "/fetch" or cmd.startswith("/fetch "):
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip().isdigit():
                    print("[Error] Usage: /fetch N  (e.g. /fetch 5)")
                    continue
                fetch_n = int(parts[1].strip())
                if fetch_n < 1:
                    print("[Error] N must be >= 1.")
                    continue
                try:
                    fetch_prompt = input("  Enter prompt > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Cancelled]")
                    continue
                if not fetch_prompt:
                    print("[Error] Prompt cannot be empty.")
                    continue
                try:
                    self.fetch_n(fetch_n, fetch_prompt)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[Error] {exc}")

            # ── /evaluate ─────────────────────────────────────────────────────
            elif cmd == "/evaluate":
                if not self.user_id:
                    print("[Error] No patient selected. Use /select_patient first.")
                    continue
                try:
                    eval_query = input("  Enter user query > ").strip()
                    eval_resp = input("  Enter generated response to evaluate > ").strip()
                    sem_count = input("  Enter semantic hit count > ").strip()
                    epi_count = input("  Enter episodic hit count > ").strip()
                    st_count = input("  Enter short-term hit count > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Cancelled]")
                    continue
                if not eval_query or not eval_resp:
                    print("[Error] Both query and response are required.")
                    continue
                try:
                    self.evaluate_response(eval_query, eval_resp, sem_count, epi_count, st_count)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[Error] {exc}")

            # ── unknown command ───────────────────────────────────────────────
            elif cmd.startswith("/"):
                print(f"[Error] Unknown command: {cmd}. Type /exit to quit.")

            # ── chat ──────────────────────────────────────────────────────────
            else:
                try:
                    self.chat(cmd)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[Error] {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = HealthcareChatbot()
    bot.run()
