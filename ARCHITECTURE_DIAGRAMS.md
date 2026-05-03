# Healthcare Memory System - Complete Architecture Diagrams

## 1. Three-Layer Memory Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HEALTHCARE MEMORY SYSTEM                          │
└──────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ SEMANTIC MEMORY LAYER (Global Medical Knowledge)                  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                    ┃
┃  Source: MedQuAD Dataset (47,000+ QA pairs)                       ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ Q: What is hypertension?                                     │ ┃
┃  │ A: Hypertension is chronic elevation of blood pressure...    │ ┃
┃  │ Focus: Cardiovascular Diseases                               │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┃                                                                    ┃
┃  Storage:                                                         ┃
┃  • Vector embeddings in Mem0's vector store                      ┃
┃  • NO user_id (global scope)                                    ┃
┃  • BM25 index for keyword search                                 ┃
┃  • Entity linking enabled                                         ┃
┃                                                                    ┃
┃  Query Pattern: search(query, filters={})  [no user_id filter]   ┃
┃                                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ EPISODIC MEMORY LAYER (Patient-Specific Health History)           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                    ┃
┃  Source: Synthea Dataset (Synthetic EHR records)                 ┃
┃                                                                    ┃
┃  • Conditions (Diagnoses)                                         ┃
┃    └─ "Patient diagnosed with Asthma on 2015-03-20"             ┃
┃                                                                    ┃
┃  • Medications (Prescriptions)                                    ┃
┃    └─ "Patient prescribed Albuterol inhaler on 2015-03-20"      ┃
┃                                                                    ┃
┃  • Encounters (Visits)                                            ┃
┃    └─ "Patient visit for respiratory symptoms on 2024-01-15"    ┃
┃                                                                    ┃
┃  Storage:                                                         ┃
┃  • Same vector store as semantic                                 ┃
┃  • HAS user_id = patient UUID (per-patient scope)               ┃
┃  • BM25 index for keyword search                                 ┃
┃  • Entity linking enabled                                         ┃
┃  • Temporal metadata preserved                                    ┃
┃                                                                    ┃
┃  Query Pattern: search(query, filters={"user_id": patient_id})  ┃
┃                                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ SHORT-TERM MEMORY LAYER (Recent Conversation Context)             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                    ┃
┃  Source: Conversation messages (added via add() calls)          ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ USER: "I've been feeling dizzy lately"                      │ ┃
┃  │ ASSISTANT: "When did this start?"                           │ ┃
┃  │ USER: "About a week ago"                                    │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┃                                                                    ┃
┃  Storage:                                                         ┃
┃  • SQLite database (mem0's built-in message table)               ┃
┃  • Per-patient session scope                                     ┃
┃  • Last K messages (default K=10)                                ┃
┃  • Automatically saved during add() calls                        ┃
┃                                                                    ┃
┃  Query Pattern: get_recent_messages(user_id, limit=K)           ┃
┃                                                                    ┃
┃  Used By:                                                         ┃
┃  • LLM extraction phase (Phase 1) for context                   ┃
┃  • NOT used in search() - only in add() pipeline               ┃
┃                                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 2. Add Flow (Memory Insertion Pipeline)

```
USER APPLICATION
    │
    ├─ User Input: "I have shortness of breath"
    │
    ├─ Determine Memory Type:
    │   ├─ If semantic knowledge: add(messages, infer=False)
    │   ├─ If patient history: add(messages, user_id=patient_id, infer=False)
    │   └─ If conversation: add(messages, user_id=patient_id, infer=True)
    │
    └─ Call: healthcare_memory_system.add_*()
            │
            └─> HealthcareMemorySystem
                    │
                    ├─ memory.add(messages, user_id, infer)
                    │
                    └─> MEM0 CORE PROCESSING
                            │
                            ├─ PHASE 1: Fetch Recent Context
                            │   └─ Query SQLite for last K messages
                            │       (used if infer=True)
                            │
                            ├─ PHASE 2: LLM Extraction (if infer=True)
                            │   ├─ Build prompt with recent context
                            │   ├─ LLM extracts: "Patient has SOB, shortness of breath symptom"
                            │   └─ Extract key entities and facts
                            │
                            ├─ PHASE 3: Batch Embedding
                            │   └─ Embed all extracted texts in single API call
                            │
                            ├─ PHASE 4: Vector Insert
                            │   ├─ Store in vector store
                            │   ├─ If user_id: scope to patient
                            │   └─ If no user_id: global scope
                            │
                            ├─ PHASE 5: Deduplication
                            │   └─ Check for similar existing memories
                            │
                            ├─ PHASE 6: Entity Linking
                            │   ├─ Extract entities: ["shortness of breath"]
                            │   ├─ Embed entities
                            │   ├─ Link to memories
                            │   └─ Add entity boost scores
                            │
                            └─ PHASE 7: SQLite History
                                └─ Save message to history table
                                   (both add and update operations)
                            
                            RESULT: Return {"added": N, "tokens_used": X}
```

---

## 3. Search Flow (Hybrid Ranking Pipeline)

```
USER QUERY: "respiratory medication symptoms"
    │
    └─> healthcare_memory_system.search_patient_knowledge(
            patient_id="uuid",
            query="respiratory medication symptoms",
            include_semantic=True
        )
        │
        ├─ STEP 1: EPISODIC SEARCH (Patient-specific)
        │   │
        │   └─> memory.search(
        │           query="respiratory medication symptoms",
        │           filters={"user_id": patient_id}  ← KEY: Patient isolation
        │       )
        │       │
        │       ├─ PREPROCESSING
        │       │   ├─ Tokenize: ["respiratory", "medication", "symptoms"]
        │       │   ├─ Lemmatize: ["respir", "medicin", "symptom"]
        │       │   └─ Create embedding: [0.15, 0.22, ..., 0.89]
        │       │
        │       ├─ DENSE SEARCH (Semantic Vectors)
        │       │   ├─ Find similar embeddings
        │       │   ├─ Apply user_id filter
        │       │   └─ Candidates: 
        │       │       • "Patient prescribed Albuterol inhaler" (0.92)
        │       │       • "Patient encounter for respiratory exam" (0.88)
        │       │       • "Patient diagnosed with Asthma" (0.81)
        │       │
        │       ├─ BM25 SEARCH (Keyword Matching)
        │       │   ├─ Find exact term matches
        │       │   ├─ Apply user_id filter
        │       │   └─ Candidates:
        │       │       • "respiratory" matches: 3 memories
        │       │       • "medication" matches: 5 memories
        │       │
        │       ├─ ENTITY BOOSTING
        │       │   ├─ Find related entities in entity store
        │       │   ├─ Add 0.0-0.5 score boost if entity matched
        │       │   └─ Attenuate by num_linked entities
        │       │
        │       ├─ HYBRID SCORING
        │       │   └─ For each memory:
        │       │       score = 0.40×semantic_score
        │       │             + 0.35×bm25_score
        │       │             + entity_boost
        │       │
        │       ├─ RANKING & FILTERING
        │       │   ├─ Sort by score (descending)
        │       │   ├─ Apply threshold (min 0.1)
        │       │   └─ Return top K
        │       │
        │       └─ EPISODIC RESULTS:
        │           [
        │               {"memory": "Patient prescribed Albuterol...", "score": 0.94},
        │               {"memory": "Patient encounter for respiratory...", "score": 0.87},
        │               {"memory": "Patient diagnosed with Asthma...", "score": 0.79}
        │           ]
        │
        ├─ STEP 2: SEMANTIC SEARCH (Global medical knowledge)
        │   │
        │   └─> memory.search(
        │           query="respiratory medication symptoms",
        │           filters={}  ← NO user_id filter
        │       )
        │       │
        │       ├─ Same preprocessing as episodic
        │       ├─ Dense + BM25 + entity boost
        │       ├─ NO user_id filtering
        │       │
        │       └─ SEMANTIC RESULTS:
        │           [
        │               {"memory": "Asthma is respiratory disease...", "score": 0.81},
        │               {"memory": "Bronchodilators treat respiratory...", "score": 0.76},
        │               {"memory": "Symptoms of asthma include...", "score": 0.72}
        │           ]
        │
        └─ RETURN:
            {
                "episodic": [...],           ← Patient-specific memories
                "semantic": [...],           ← General medical knowledge
                "combined_count": 6,
                "search_time_ms": 245
            }
```

---

## 4. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                              │
│  (Your code using the healthcare memory system)                    │
└─────────────────────────────────────────────────────────────────────┘
            │                           │                      │
            ▼                           ▼                      ▼
┌──────────────────────┐    ┌──────────────────────┐  ┌──────────────┐
│ DATASET LOADER       │    │ HEALTHCARE MEMORY    │  │ VALIDATION   │
│ ──────────────────   │    │ SYSTEM               │  │ ──────────── │
│                      │    │ ───────────────────  │  │              │
│ healthcare_          │    │ • add_semantic()     │  │ • validate_  │
│ integration.py       │    │ • add_patient()      │  │   semantic() │
│ ──────────────────   │    │ • search_patient()   │  │ • validate_  │
│                      │    │ • get_history()      │  │   episodic() │
│ • load_medquad()     │    │ • add_context()      │  │ • etc.       │
│ • load_synthea_*()   │    │ • get_context()      │  │              │
│ • format_*()         │    │                      │  │              │
└──────────────────────┘    └──────────┬───────────┘  └──────────────┘
        │                              │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │    MEM0 MEMORY CLASS         │
        │  (mem0/memory/main.py)       │
        ├──────────────────────────────┤
        │                              │
        │ PUBLIC API:                  │
        │  • add(messages, ...)        │
        │  • search(query, filters)    │
        │  • get_recent_messages()     │
        │  • update/delete             │
        │                              │
        └────────────┬─────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ VECTOR  │ │  ENTITY │ │ SQLite  │
    │ STORE   │ │  STORE  │ │ Database│
    ├─────────┤ ├─────────┤ ├─────────┤
    │ Qdrant/ │ │ Entity  │ │ Messages│
    │ Pinecone│ │ vectors │ │ History │
    │ etc.    │ │ &links  │ │ & audit │
    │         │ │         │ │         │
    │ Memories│ │ Entity  │ │ Recent  │
    │ with    │ │ boost   │ │ context │
    │ user_id │ │ scores  │ │ & trail │
    └─────────┘ └─────────┘ └─────────┘

SCOPING RULES:

SEMANTIC MEMORY:
  └─ NO user_id in metadata
     • Vector Store: Global
     • Query: filters={}
     • Access: All patients can see
     • Example: Medical knowledge

EPISODIC MEMORY:
  └─ HAS user_id = patient_uuid
     • Vector Store: Scoped by user_id
     • Query: filters={"user_id": patient_id}
     • Access: Only that patient's data
     • Example: Patient history

SHORT-TERM MEMORY:
  └─ SQLite with user_id scoping
     • Database: Scoped by session_scope
     • Query: Session scope includes patient_id
     • Access: Only recent messages for that patient
     • Example: Recent conversation
```

---

## 5. Parallel Operations Timeline

```
PARALLEL OPERATIONS IN ADD PIPELINE

User calls: memory.add(messages, user_id=patient_id, infer=True)
│
├─ [START] ─────────────────────────────────────────────────────┐
│                                                                │
├─ [PHASE 1] Fetch Context (can be parallel)                   │
│  ├─ Query 1: Get recent messages from SQLite  ┐              │
│  └─ Query 2: Search for related memories ──┬─┘              │
│                                             │ (PARALLEL)     │
├─ [PHASE 2] LLM Extraction (waits for Phase 1)                │
│  └─ Call LLM with context + messages ───────┬──────┐         │
│                                             │      │         │
├─ [PHASE 3] Batch Embedding (parallel API call)               │
│  ├─ Embed extracted fact 1  ┐                                │
│  ├─ Embed extracted fact 2  ├─> Single API call (all at once)│
│  └─ Embed extracted fact N  ┘                                │
│                                                                │
├─ [PHASE 4] Vector Insert                                      │
│  └─ Insert all embeddings to vector store                     │
│                                                                │
├─ [PHASE 5] Deduplication                                      │
│  └─ Check for similar existing memories                       │
│                                                                │
├─ [PHASE 6] Entity Linking (can be parallel)                   │
│  ├─ Extract entities  ┐                                       │
│  ├─ Embed entities    ├─> Parallel steps                     │
│  └─ Search + link ────┘                                       │
│                                                                │
├─ [PHASE 7] SQLite History                                     │
│  └─ Save to history table                                     │
│                                                                │
└─ [END] ──────────────────────────────────────────────────────┘

PARALLEL OPERATIONS IN SEARCH PIPELINE

User calls: memory.search(query, filters={"user_id": patient_id})
│
├─ [START] ──────────────────────────────────────────────────┐
│                                                             │
├─ [PREPROCESSING]                                            │
│  └─ Tokenize, lemmatize, embed query                       │
│                                                             │
├─ [DUAL SEARCH] (can be parallel)                           │
│  ├─ Dense search (vector similarity)  ┐                    │
│  └─ BM25 search (keyword matching) ───┼─> Parallel         │
│                                        │                    │
├─ [SCORING]                                                  │
│  └─ Combine scores from both searches                      │
│                                                             │
├─ [ENTITY BOOST] (can be parallel)                          │
│  ├─ Extract entities from top results  ┐                   │
│  ├─ Search entity store  ───────────────┼─> Parallel       │
│  └─ Apply boost scores ─────────────────┘                  │
│                                                             │
├─ [RANKING & FILTERING]                                     │
│  ├─ Sort by final score                                    │
│  ├─ Apply user_id filter ← ENFORCED                        │
│  └─ Return top K                                            │
│                                                             │
└─ [END] ───────────────────────────────────────────────────┘
```

---

## 6. Patient Isolation Guarantee

```
MEMORY STORE LAYOUT

Vector Store (same collection):
┌────────────────────────────────────────────────────────┐
│ Memory 1: "Asthma is..." (metadata: user_id=null)     │ ← SEMANTIC
│ Memory 2: "Hypertension is..." (metadata: user_id=null)│ ← SEMANTIC
│ Memory 3: "Patient A diagnosed asthma..." (user_id=A)  │ ← EPISODIC
│ Memory 4: "Patient A prescribed Albuterol..." (user_id=A)│ ← EPISODIC
│ Memory 5: "Patient B diagnosed diabetes..." (user_id=B) │ ← EPISODIC
│ Memory 6: "Patient B prescribed Metformin..." (user_id=B)│ ← EPISODIC
└────────────────────────────────────────────────────────┘

ISOLATION BY FILTERING

Query from Patient A:
  search(query, filters={"user_id": "A"})
    └─ Returns: Memory 3, Memory 4, Memory 1, Memory 2
       (Patient A's memories + semantic knowledge)

Query from Patient B:
  search(query, filters={"user_id": "B"})
    └─ Returns: Memory 5, Memory 6, Memory 1, Memory 2
       (Patient B's memories + semantic knowledge)

Query for Semantic Only:
  search(query, filters={})
    └─ Returns: Memory 1, Memory 2 (no user_id)
       (Global medical knowledge only)

    Patient A sees:
    ├─ Their data (user_id="A") ✓
    ├─ Global semantic (user_id=null) ✓
    └─ Patient B's data (user_id="B") ✗ BLOCKED by filter

    Patient B sees:
    ├─ Their data (user_id="B") ✓
    ├─ Global semantic (user_id=null) ✓
    └─ Patient A's data (user_id="A") ✗ BLOCKED by filter
```

---

## 7. Semantic vs Episodic Storage Comparison

```
STORAGE COMPARISON

                    SEMANTIC              EPISODIC            SHORT-TERM
────────────────────────────────────────────────────────────────────────────
Store Type          Vector Store          Vector Store        SQLite DB
                    (Qdrant/Pinecone)     (same as semantic)  (messages table)

Scope               None (global)         user_id (patient)   session_scope
                                                              (includes user_id)

Example             "Asthma is a          "Patient diagnosed  "User: I feel dizzy"
                    chronic respiratory   Asthma on 2015..."  "Assistant: When?"
                    disease..."

Query Filter        filters={}            filters=            Session scope
                    (no filtering)        {"user_id":         filtering in SQLite
                                         patient_id}

User_id in Meta     NO                    YES                 YES (implicit)
────────────────────────────────────────────────────────────────────────────

Semantic Query Pattern:
  results = memory.search("diabetes treatment", filters={})
  └─ Returns global medical knowledge only

Episodic Query Pattern:
  results = memory.search("patient symptoms", filters={"user_id": patient_id})
  └─ Returns patient A's data + semantic knowledge (filtered by patient)

Short-term Query Pattern:
  messages = memory.get_recent_messages(user_id=patient_id, limit=10)
  └─ Returns last 10 messages for patient A's session from SQLite
```

---

## 8. Complete End-to-End Example

```
APPLICATION FLOW: Healthcare Q&A for Patient

Step 1: SETUP
  system = HealthcareMemorySystem()
  system.add_semantic_medical_knowledge(limit=1000)
    └─ Loads MedQuAD into vector store (no user_id)

Step 2: ADD PATIENT
  patient_id = "a1b2c3d4..."
  system.add_patient_history(patient_id)
    └─ Loads Synthea into vector store (with user_id=patient_id)

Step 3: PATIENT CONVERSATION BEGINS
  User: "I feel shortness of breath, is this serious?"

Step 4: ADD TO SHORT-TERM MEMORY
  system.add_patient_short_term_context(patient_id, [
    {"role": "user", "content": "I feel shortness of breath..."}
  ])
    ├─ Phase 1: Fetch recent context (empty on first call)
    ├─ Phase 2: LLM extracts: "Patient experiencing SOB, concern about severity"
    ├─ Phase 3: Embed extracted facts
    ├─ Phase 4: Insert to vector store (with user_id=patient_id) ← EPISODIC
    └─ Phase 5: Save to SQLite

Step 5: SEARCH ALL MEMORIES
  results = system.search_patient_knowledge(
    patient_id=patient_id,
    query="shortness of breath causes symptoms",
    include_semantic=True
  )

  Episodic Search Results (patient-specific):
    ├─ "Patient diagnosed Asthma on 2015..." (score: 0.94)
    └─ "Patient prescribed Albuterol inhaler" (score: 0.87)

  Semantic Search Results (medical knowledge):
    ├─ "Asthma is a respiratory disease characterized by..." (score: 0.81)
    ├─ "Common causes of shortness of breath..." (score: 0.76)
    └─ "When to seek emergency care for SOB..." (score: 0.72)

Step 6: COMBINE FOR LLM
  LLM receives:
    ├─ Patient's recent symptoms (short-term)
    ├─ Patient's medical history (episodic)
    └─ General medical knowledge (semantic)
    
  LLM Response:
    "Based on your history of asthma and current symptoms,
     this could be an asthma exacerbation. Given your
     recent albuterol use, you should..."

Step 7: SAVE RESPONSE TO SHORT-TERM
  system.add_patient_short_term_context(patient_id, [
    {"role": "assistant", "content": "Based on your history..."}
  ])
    └─ Next query will have even more context

Result: Complete semantic + episodic + short-term memory integration
```

---

This comprehensive architecture enables:

✓ **Semantic/Episodic Separation** - Clear distinction between global knowledge and patient data
✓ **Patient Isolation** - Strong guarantees that Patient A cannot see Patient B's data
✓ **Hybrid Search** - Combines all memory types with intelligent ranking
✓ **Short-term Context** - Immediate conversation history for better extractions
✓ **Scalability** - Parallel operations where possible
✓ **Healthcare-Specific** - Temporal data, entity linking, clinical terminology support
