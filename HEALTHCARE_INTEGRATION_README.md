# Healthcare Memory System Integration Guide

## Overview

This integration combines **MedQuAD** (semantic medical knowledge) and **Synthea** (episodic patient histories) into Mem0's memory architecture, demonstrating proper separation of:

1. **SEMANTIC MEMORY** (MedQuAD + Flashcards): Global medical knowledge, shared across all patients
2. **EPISODIC MEMORY** (Synthea): Patient-specific histories, scoped to individual patients
3. **SHORT-TERM MEMORY** (SQLite): Recent conversation context for each patient

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              HEALTHCARE MEMORY SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEMANTIC MEMORY LAYER (Vector Store)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MedQuAD & Flashcards: Medical Knowledge Base              │   │
│  │ • Global, no user_id                                     │   │
│  │ • 47,000+ MedQuAD QA pairs + Medical Flashcards          │   │
│  │ • Example: "Glaucoma is a disease that damages..."      │   │
│  │ • Storage: Vector embeddings (no scope filtering)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  EPISODIC MEMORY LAYER (Vector Store)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Synthea: Patient Health Histories                        │   │
│  │ • Per-patient (scoped to user_id = patient UUID)        │   │
│  │ • Conditions, medications, encounters                    │   │
│  │ • Example: "Patient diagnosed with bronchitis 2013..."  │   │
│  │ • Storage: Vector embeddings (WITH user_id filter)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  SHORT-TERM MEMORY LAYER (SQLite)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Recent Conversation History                              │   │
│  │ • Last K messages per patient session                    │   │
│  │ • Immediate context for LLM extraction                  │   │
│  │ • Per-patient scoping                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files

### 1. `healthcare_integration.py`
**Dataset loader for MedQuAD and Synthea data**

```python
from healthcare_integration import HealthcareDatasetLoader

loader = HealthcareDatasetLoader()

# Load SEMANTIC knowledge (MedQuAD)
medquad_list = loader.load_medquad(limit=100)

# Load EPISODIC patient data (Synthea)
patients = loader.load_synthea_patients()
conditions = loader.load_synthea_conditions(patient_id="uuid")
medications = loader.load_synthea_medications(patient_id="uuid")
encounters = loader.load_synthea_encounters(patient_id="uuid")

# Format for Mem0
semantic_formatted = loader.format_semantic_memory(qa)
episodic_formatted = loader.format_episodic_condition(condition)
```

**Functions:**
- `load_medquad()` - Load medical QA pairs
- `load_medical_flashcards()` - Load medical flashcards
- `load_synthea_patients()` - Load patient demographics
- `load_synthea_conditions()` - Load diagnoses (per patient)
- `load_synthea_medications()` - Load prescriptions (per patient)
- `load_synthea_encounters()` - Load visits (per patient)
- `format_*()` - Convert to memory statements

---

### 2. `healthcare_memory_system.py`
**Complete integration system**

```python
from healthcare_memory_system import HealthcareMemorySystem

system = HealthcareMemorySystem()

# Step 1: Add SEMANTIC medical knowledge (global)
semantic_result = system.add_semantic_medical_knowledge(limit=1000)

# Step 2: Add EPISODIC patient history
patient_id = "some-patient-uuid"
episodic_result = system.add_patient_history(patient_id)

# Step 3: Add SHORT-TERM conversation context
conversation = [
    {"role": "user", "content": "I have shortness of breath"},
    {"role": "assistant", "content": "When did this start?"}
]
system.add_patient_short_term_context(patient_id, conversation)

# Step 4: Search across all memory types
results = system.search_patient_knowledge(
    patient_id=patient_id,
    query="respiratory symptoms",
    include_semantic=True
)

# Step 5: Retrieve patient history
history = system.get_patient_full_history(patient_id)
```

**Key Methods:**
- `add_semantic_medical_knowledge()` - Load MedQuAD + Flashcards globally
- `add_patient_history()` - Load Synthea data for a patient
- `add_patient_short_term_context()` - Add recent messages
- `search_patient_knowledge()` - Hybrid search (semantic + episodic)
- `get_patient_full_history()` - Get all patient memories
- `get_patient_recent_context()` - Get recent messages

---

### 3. `test_healthcare_integration.py`
**Comprehensive test suite**

```bash
pytest test_healthcare_integration.py -v
```

**Test Classes:**
- `TestMemorySeparation` - Verify semantic/episodic are separate
- `TestShortTermMemory` - Verify SQLite context retrieval
- `TestHybridSearch` - Verify search combines both memory types
- `TestDataIntegrity` - Verify operations preserve scope

**Key Tests:**
- `test_semantic_memory_has_no_user_id()` - Semantic must not be scoped
- `test_episodic_memory_scoped_to_user()` - Episodic must be scoped
- `test_short_term_messages_stored()` - Messages saved to SQLite
- `test_short_term_memory_scoped_per_patient()` - Per-patient isolation

---

### 4. `validate_architecture.py`
**Architecture validation script**

```bash
python validate_architecture.py
```

**Validations:**
1. **Semantic vs Episodic Separation** - Verify proper scoping
2. **Short-Term Memory** - Verify SQLite retrieval
3. **Memory Scoping** - Verify user_id filtering works
4. **Hybrid Search** - Verify combined ranking
5. **Entity Linking** - Verify entity boosts work

---

## Memory Separation Details

### SEMANTIC MEMORY (MedQuAD)

**Characteristics:**
- Global knowledge, shared across all patients
- NO `user_id` in storage
- NOT filtered by patient
- Example: "Hypertension is chronic elevation of blood pressure"

**Storage:**
```python
memory.add(
    messages=[{"role": "user", "content": "Hypertension definition..."}],
    # NO user_id here
    infer=False
)
```

**Retrieval:**
```python
# Can be searched globally (no user_id filter needed)
results = memory.search(
    query="hypertension",
    filters={},  # No patient scope
    top_k=10
)
```

### EPISODIC MEMORY (Synthea)

**Characteristics:**
- Patient-specific health history
- MUST have `user_id` = patient UUID
- Filtered by patient
- Example: "Patient diagnosed with bronchitis on 2013-06-24"

**Storage:**
```python
memory.add(
    messages=[{"role": "user", "content": "Patient diagnosed with..."}],
    user_id="patient-uuid",  # ← REQUIRED
    infer=False
)
```

**Retrieval:**
```python
# Must be searched with user_id filter
results = memory.search(
    query="bronchitis",
    filters={"user_id": "patient-uuid"},  # ← Required filter
    top_k=10
)
```

### SHORT-TERM MEMORY (SQLite)

**Characteristics:**
- Recent conversation messages
- Per-patient scoping
- Automatically saved during `add()`
- Used for LLM context in extraction phase

**Storage:**
```python
memory.add(
    messages=[
        {"role": "user", "content": "I feel dizzy"},
        {"role": "assistant", "content": "When did this start?"}
    ],
    user_id="patient-uuid"
)
# Messages automatically saved to SQLite
```

**Retrieval:**
```python
recent = memory.get_recent_messages(
    user_id="patient-uuid",
    limit=10
)
# Returns last 10 messages as context
```

---

## Hybrid Search Architecture

When searching for patient information, Mem0 uses **4-layer scoring**:

```
User Query: "respiratory symptoms medication"
    ↓
[PREPROCESSING]
  ├─ Lemmatize: "respir symptom medicin"
  ├─ Extract entities: ["respiratory", "medication"]
  └─ Create embedding: [0.15, 0.22, ..., 0.89]

    ↓
[DUAL SEARCH]
  ├─ Semantic Search (Dense Vectors)
  │   └─ Find embeddings similar to query
  │       ├─ SEMANTIC: "Asthma is respiratory disease" (0.92)
  │       └─ EPISODIC: "Patient has asthma, prescribed Albuterol" (0.88)
  │
  └─ Keyword Search (BM25)
      └─ Find exact term matches
          ├─ SEMANTIC: Medical articles on respiratory meds
          └─ EPISODIC: Patient's medication records

    ↓
[SCORING]
  ├─ Semantic score (dense similarity): 0-1 range
  ├─ BM25 score (normalized): 0-1 range
  ├─ Entity boost: +0.0-0.5 if entity matched
  └─ Final formula: 0.4×semantic + 0.35×bm25 + entity_boost

    ↓
[RANKING & FILTERING]
  ├─ Sort by final score
  ├─ Apply threshold (min 0.1)
  ├─ Filter by user_id (episodic only)
  └─ Return top K

    ↓
Results:
  1. "Patient prescribed Albuterol inhaler for asthma" (Score: 0.94)
  2. "Patient diagnosed with asthma on 2015-03-20" (Score: 0.89)
  3. "Asthma is chronic respiratory disease..." (Score: 0.81)
```

---

## Usage Examples

### Example 1: Load Medical Knowledge Base

```python
from healthcare_memory_system import HealthcareMemorySystem

system = HealthcareMemorySystem()

# Load 1000 medical QA pairs from MedQuAD
result = system.add_semantic_medical_knowledge(limit=1000)
print(f"Loaded {result['added']} medical facts")
# Output: Loaded 1000 medical facts
```

### Example 2: Add Patient History

```python
from healthcare_memory_system import HealthcareMemorySystem

system = HealthcareMemorySystem()

patient_id = "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6"

# Load patient's complete history from Synthea
result = system.add_patient_history(patient_id)
print(f"Added for patient {patient_id}:")
print(f"  - Conditions: {result['conditions']}")
print(f"  - Medications: {result['medications']}")
print(f"  - Encounters: {result['encounters']}")
```

### Example 3: Add Conversation Context

```python
system.add_patient_short_term_context(
    patient_id=patient_id,
    conversation=[
        {"role": "user", "content": "My blood pressure is 160/100"},
        {"role": "assistant", "content": "That's elevated. Have you been diagnosed with hypertension?"},
        {"role": "user", "content": "Yes, my doctor prescribed Lisinopril last year"}
    ]
)
```

### Example 4: Search Patient Knowledge

```python
# Hybrid search combining semantic + episodic
results = system.search_patient_knowledge(
    patient_id=patient_id,
    query="blood pressure medication hypertension",
    top_k=10,
    include_semantic=True
)

print(f"Found {len(results['episodic'])} episodic (patient-specific) results")
print(f"Found {len(results['semantic'])} semantic (general knowledge) results")

for mem in results['episodic']:
    print(f"  - {mem['memory']} (score: {mem['score']:.2f})")
```

---

## Dataset Information

### MedQuAD & Flashcards (Semantic Memory)

- **MedQuAD**: NIH National Institute of Health (47,000+ QA pairs)
- **Medical Flashcards**: Semantic medical question-answer flashcards
- **Coverage**: Diseases, drugs, side effects, therapies, treatments
- **Usage**: Global medical knowledge base
- **Scoping**: None (global, shared)

### Synthea (Episodic Memory)

- **Source**: Synthea (synthetic patient generator)
- **Type**: Realistic synthetic patient EHR records
- **Includes**:
  - Patient demographics
  - Encounters (visits)
  - Conditions (diagnoses)
  - Medications (prescriptions)
  - Immunizations
  - Procedures
  - Observations
- **Usage**: Patient health histories
- **Scoping**: By patient ID (user_id)

---

## Architecture Validation

Run the validation script to verify the architecture:

```bash
python validate_architecture.py
```

**Expected Output:**
```
✓ PASS: semantic_episodic_separation
✓ PASS: short_term_memory
✓ PASS: memory_scoping
✓ PASS: hybrid_search
✓ PASS: entity_linking

✓ ALL TESTS PASSED - Architecture is correctly implemented!
```

---

## Key Guarantees

### 1. Semantic/Episodic Separation ✓
- Semantic memories (MedQuAD) have NO user_id
- Episodic memories (Synthea) HAVE user_id
- They are stored in same vector store but queried separately

### 2. Patient Isolation ✓
- Patient A's memories are NOT visible to Patient B
- Enforced via `filters={"user_id": patient_id}` in all queries
- Short-term memory SQLite also scoped per patient

### 3. Short-Term Context ✓
- Recent messages saved automatically during `add()`
- Available via `get_recent_messages(user_id=patient_id, limit=K)`
- Used by LLM during extraction phase (Phase 1)

### 4. Hybrid Search ✓
- Combines semantic vectors + BM25 keyword search + entity boosts
- Episodic results scoped to patient
- Semantic results global but can be filtered post-hoc
- Ranking formula: `0.4×semantic + 0.35×bm25 + entity_boost`

---

## Troubleshooting

### Issue: Semantic memories appearing in patient search

**Cause**: Both semantic and episodic stored in same vector store

**Solution**: 
- Semantic memories intentionally added without user_id
- Search with `filters={"user_id": patient_id}` only finds episodic
- To include semantic in patient search, search globally then filter post-hoc

### Issue: Short-term memory not retrieving messages

**Cause**: Messages table in SQLite not being populated

**Solution**:
- Ensure `add()` is called with `user_id` parameter
- Check SQLite database at `~/.mem0_data/history.db`
- Call `memory.get_recent_messages(user_id=patient_id, limit=10)`

### Issue: Low relevance scores for episodic memories

**Cause**: BM25/semantic scoring depends on embedding quality

**Solution**:
- Use more specific queries
- Add metadata to memories for better context
- Consider reranking with `rerank=True` parameter

---

## Next Steps

1. **Run Validation**: Execute `validate_architecture.py` to confirm setup
2. **Load Data**: Use `healthcare_memory_system.py` to load MedQuAD + Synthea
3. **Test Integration**: Run `test_healthcare_integration.py` with pytest
4. **Build Application**: Use `HealthcareMemorySystem` class in your application

---

## References

- **Mem0 Documentation**: https://docs.mem0.ai
- **MedQuAD**: https://github.com/abachaa/MedQuAD
- **Synthea**: https://github.com/synthetichealth/synthea
- **Architecture Analysis**: See `MEM0_WORKFLOW_ANALYSIS.md`

---

## Summary

This integration demonstrates how Mem0 can handle complex multi-memory systems:

```
SEMANTIC MEMORY       EPISODIC MEMORY      SHORT-TERM MEMORY
(MedQuAD)            (Synthea)            (SQLite)
   Global                Per-Patient          Per-Patient
   No scope              user_id              user_id
   Shared               Isolated             Recent
   Vector Store         Vector Store         Database
```

All three layers work together in **hybrid search** to provide personalized medical information that combines:
- General medical knowledge (semantic)
- Patient-specific history (episodic)
- Recent context (short-term)
