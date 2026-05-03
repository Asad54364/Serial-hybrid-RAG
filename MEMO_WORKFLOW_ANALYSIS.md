# Mem0 Memory System: Complete Workflow Analysis

## Executive Summary

Mem0 is a **multi-layer memory system** that manages AI agent memories through a phased pipeline. It handles three memory types (Semantic, Episodic, Procedural) across **two distinct processing layers**:

1. **User/Client Side** - Where memories are extracted from conversation
2. **AI LLM Side** - Where memories are interpreted and integrated into prompts

---

## 1. MEMORY TYPES

```python
# From mem0/configs/enums.py
class MemoryType:
    SEMANTIC = "semantic_memory"      # Facts, preferences (stored in vector DB)
    EPISODIC = "episodic_memory"      # Events, experiences (stored in vector DB) 
    PROCEDURAL = "procedural_memory"  # Agent behaviors, patterns (LLM-summarized)
```

### Key Difference:
- **Semantic + Episodic**: Extracted from user messages, stored directly in vector store with embeddings
- **Procedural**: Only for agents (agent_id present); LLM creates a summary of agent execution history

---

## 2. REQUEST-TO-RESPONSE PIPELINE

### A. PHASE 0-1: CLIENT PREPARES REQUEST (User/Device Side)

**What Happens on User Side:**

```python
# Client initiates (mem0/memory/main.py:577-664)
memory.add(
    messages=[{"role": "user", "content": "I like Python and traveled to Japan last month"}],
    user_id="user123",
    agent_id="agent_ai",  # Optional; triggers agent memory extraction if present
    metadata={"session": "chat_001"},
    infer=True  # If False, stores raw message; if True, LLM extracts facts
)
```

**Processing:**
1. **Validation** → Normalize entity IDs (user_id, agent_id, run_id)
2. **Message Parsing** → Convert string/dict to list of message objects
3. **Session Scoping** → Build deterministic session scope: `"user_id=user123&agent_id=agent_ai"`
4. **Vision Handling** → If `enable_vision=true`, parse images in messages

---

### B. PHASE 1: EXISTING MEMORY RETRIEVAL (Embedding Search)

**What Happens (Still on User/Client Infrastructure):**

```python
# Line 710-718 in main.py (_add_to_vector_store)
session_scope = _build_session_scope(filters)
last_messages = self.db.get_last_messages(session_scope, limit=10)  # SHORT-TERM MEMORY

parsed_messages = parse_messages(messages)  # Extract text from message list

search_filters = {"user_id": "user123", "agent_id": "agent_ai"}
query_embedding = self.embedding_model.embed(parsed_messages, "search")  # SEMANTIC SEARCH

existing_results = self.vector_store.search(
    query=parsed_messages,
    vectors=query_embedding,
    top_k=10,
    filters=search_filters
)  # Returns up to 10 existing memories
```

**Three Memory Sources Being Queried:**

| Memory Type | Source | Format | Use |
|-------------|--------|--------|-----|
| **Semantic** | Vector Store | Embeddings of facts | Find existing similar facts to avoid duplicates |
| **Episodic** | Vector Store | Embeddings of events | Retrieve past experiences for context |
| **Short-Term** | SQLite DB | Last 10 raw messages | Provide immediate context to LLM extractor |

**Parallel Operations:**
- Database query for last messages & vector search for existing memories happen **independently** (can be parallelized)

---

### C. PHASE 2: LLM EXTRACTION (AI LLM Side - First LLM Call)

**What Happens (Sent to AI LLM):**

```python
# Line 727-750 in main.py
system_prompt = ADDITIVE_EXTRACTION_PROMPT  
# From mem0/configs/prompts.py:468-625

is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
if is_agent_scoped:
    system_prompt += AGENT_CONTEXT_SUFFIX  # Frame memories from agent perspective

user_prompt = generate_additive_extraction_prompt(
    existing_memories=[
        {"id": "0", "text": "User works as a Software Engineer"},
        {"id": "1", "text": "User likes Python programming"}
    ],
    new_messages=[
        {"role": "user", "content": "I like Python and traveled to Japan last month"}
    ],
    last_k_messages=[
        {"role": "user", "message": "Hi, how are you?"},
        {"role": "assistant", "message": "I'm doing great!"}
    ],
    custom_instructions=metadata.get("instructions")
)

# LLM Call #1
response = self.llm.generate_response(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format={"type": "json_object"}
)
```

**LLM Receives (In Prompt):**

```
## Summary
[Empty initially, or from get_recent_messages()]

## Last k Messages  
user: Hi, how are you?
assistant: I'm doing great!

## Recently Extracted Memories
[]

## Existing Memories
[
  {"id": "0", "text": "User works as a Software Engineer"},
  {"id": "1", "text": "User likes Python programming"}
]

## New Messages
[{"role": "user", "content": "I like Python and traveled to Japan last month"}]

## Observation Date
2025-05-02

## Current Date
2025-05-02

# Output:
```

**LLM Returns (JSON):**

```json
{
  "memory": [
    {
      "id": "0",
      "text": "User traveled to Japan last month",
      "attributed_to": "user",
      "linked_memory_ids": []
    },
    {
      "id": "1", 
      "text": "User prefers Python programming",
      "attributed_to": "user",
      "linked_memory_ids": ["1"]  // Links to existing memory about Python
    }
  ]
}
```

**Key Points:**
- LLM sees **only** existing memories (not raw vectors)
- Returns **only NEW memories to ADD** (not updates/deletes in this flow)
- Can link to existing memories via `linked_memory_ids`
- **NO information about what the agent/AI said** is stored here (only user facts)

---

### D. PHASE 3: BATCH EMBEDDING (User/Client Side)

**What Happens:**

```python
# Line 774-787 in main.py
mem_texts = [
    "User traveled to Japan last month",
    "User prefers Python programming"
]

# Batch embed all extracted memories
try:
    mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
    embed_map = {
        "User traveled to Japan last month": [0.15, 0.22, ..., 0.89],
        "User prefers Python programming": [0.18, 0.20, ..., 0.91]
    }
except:
    # Fallback: embed one by one
    embed_map = {}
    for text in mem_texts:
        embed_map[text] = self.embedding_model.embed(text, "add")
```

**Parallelization:** All embeddings computed in **single batch call** to embedding API (parallel).

---

### E. PHASE 4-5: DEDUPLICATION & CPU PROCESSING (User/Client Side)

**What Happens:**

```python
# Line 788-807 in main.py
existing_hashes = set()
for mem in existing_results:
    h = mem.payload.get("hash")  # MD5 of memory text
    if h:
        existing_hashes.add(h)

records = []  # (memory_id, text, embedding, payload)
seen_hashes = set()

for mem in extracted_memories:
    text = mem.get("text")
    mem_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Skip if already in vector store or batch
    if mem_hash in existing_hashes or mem_hash in seen_hashes:
        logger.debug(f"Skipping duplicate: {text[:50]}")
        continue
    
    seen_hashes.add(mem_hash)
    
    # Lemmatize for BM25 keyword search
    text_lemmatized = lemmatize_for_bm25(text)
    
    memory_id = str(uuid.uuid4())
    mem_metadata = {
        "data": text,
        "text_lemmatized": text_lemmatized,
        "hash": mem_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": created_at,
        "attributed_to": mem.get("attributed_to", "user"),
        "user_id": filters["user_id"],
        "agent_id": filters.get("agent_id"),
        "run_id": filters.get("run_id")
    }
    
    records.append((memory_id, text, embed_map[text], mem_metadata))
```

**Operations (All on Client Side):**
- Hash-based deduplication (MD5)
- Lemmatization for BM25 (keyword search index)
- Metadata construction
- Record batching for bulk insert

---

### F. PHASE 6: BATCH PERSIST TO VECTOR STORE (User/Client Side)

**What Happens:**

```python
# Line 828-845 in main.py
all_vectors = [r[2] for r in records]      # All embeddings
all_ids = [r[0] for r in records]          # All memory IDs
all_payloads = [r[3] for r in records]     # All metadata

try:
    self.vector_store.insert(
        vectors=all_vectors,
        ids=all_ids,
        payloads=all_payloads
    )
    # Single batch insert to vector DB (Qdrant, Pinecone, etc.)
except:
    # Fallback: insert one by one
    for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
        self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
```

**What Gets Stored in Vector Store:**
```python
{
    "id": "a1b2c3d4-...",  # UUID
    "vector": [0.15, 0.22, ..., 0.89],  # Embedding
    "payload": {
        "data": "User traveled to Japan last month",
        "text_lemmatized": "user travel japan month",  # For BM25
        "hash": "abc123def456...",
        "created_at": "2025-05-02T19:45:40.371+05:00",
        "updated_at": "2025-05-02T19:45:40.371+05:00",
        "attributed_to": "user",
        "user_id": "user123",
        "agent_id": "agent_ai"
    }
}
```

---

### G. PHASE 7: BATCH ENTITY EXTRACTION & LINKING (User/Client Side)

**What Happens:**

```python
# Line 869-959 in main.py
all_texts = [r[1] for r in records]  # All memory texts

# 7a: Extract all entities from all texts (NER - Named Entity Recognition)
try:
    all_entities = extract_entities_batch(all_texts)
    # Returns: [
    #   [(PERSON, "Japan"), (LOCATION, "Tokyo")],
    #   [(PROGRAMMING_LANG, "Python")]
    # ]
except Exception:
    all_entities = []

# 7b: Global dedup
global_entities = {}  # normalized_key -> (type, text, set of memory_ids)
for idx, (memory_id, text, embedding, payload) in enumerate(records):
    entities = all_entities[idx] if idx < len(all_entities) else []
    for entity_type, entity_text in entities:
        key = entity_text.strip().lower()
        if key in global_entities:
            global_entities[key][2].add(memory_id)  # Add memory_id to set
        else:
            global_entities[key] = [entity_type, entity_text, {memory_id}]

# 7c: Batch embed all unique entities
entity_texts = [global_entities[k][1] for k in global_entities.keys()]
try:
    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
    # Parallel embedding of all entities
except:
    entity_embeddings = [None] * len(entity_texts)  # Fallback

# 7d: Batch search for existing entities
valid_vectors = [entity_embeddings[i] for i in valid_indices]
existing_matches = self.entity_store.search_batch(
    queries=valid_texts,
    vectors_list=valid_vectors,
    top_k=1,
    filters={"user_id": "user123", "agent_id": "agent_ai"}
)
# Returns existing entity matches or None

# 7e: Separate into INSERT vs UPDATE
to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
for j, key in enumerate(valid_keys):
    matches = existing_matches[j] if j < len(existing_matches) else []
    
    if matches and matches[0].score >= 0.95:
        # Entity already exists - UPDATE it
        match = matches[0]
        payload = match.payload or {}
        linked = set(payload.get("linked_memory_ids", []))
        linked.update(global_entities[key][2])  # Add new memory_ids
        payload["linked_memory_ids"] = sorted(linked)
        self.entity_store.update(
            vector_id=match.id,
            vector=None,
            payload=payload
        )
    else:
        # New entity - queue for INSERT
        to_insert_vectors.append(valid_vectors[j])
        to_insert_ids.append(str(uuid.uuid4()))
        to_insert_payloads.append({
            "data": entity_text,
            "entity_type": entity_type,
            "linked_memory_ids": sorted(global_entities[key][2]),
            "user_id": "user123",
            "agent_id": "agent_ai"
        })

# 7f: Batch insert all new entities
if to_insert_vectors:
    self.entity_store.insert(
        vectors=to_insert_vectors,
        ids=to_insert_ids,
        payloads=to_insert_payloads
    )
```

**Entity Store Schema:**
```python
{
    "id": "entity_uuid_...",
    "vector": [0.12, 0.34, ..., 0.78],  # Embedding of entity text
    "payload": {
        "data": "Japan",
        "entity_type": "LOCATION",
        "linked_memory_ids": ["a1b2c3d4-...", "e5f6g7h8-..."],
        "user_id": "user123",
        "agent_id": "agent_ai"
    }
}
```

**Why Entities Matter:**
- **Semantic Boost During Search**: When searching for "I went to Japan", find "Japan" entity with 95%+ similarity → boost all linked memories
- **Knowledge Graph**: Build connections: `Japan ←→ [travel_memory_1, travel_memory_2, culture_memory_3]`
- **Fast Entity Queries**: "Find all memories about Japan" - search entity store first

---

### H. PHASE 8: PERSIST SHORT-TERM MESSAGES (SQLite)

**What Happens:**

```python
# Line 962 in main.py
self.db.save_messages(messages, session_scope)

# In storage.py:_create_messages_table()
# Stores raw messages in SQLite for quick recall
INSERT INTO messages (id, content, role, session_scope, created_at)
VALUES (
    "msg_uuid",
    "I like Python and traveled to Japan last month",
    "user",
    "user_id=user123&agent_id=agent_ai",
    "2025-05-02T19:45:40.371+05:00"
)
```

**Purpose:**
- **Short-term/Working Memory**: Quick access to last K messages
- Used in Phase 1 for LLM context
- Kept separate from long-term (vector store) for speed

---

### I. PHASE 9: HISTORY TRACKING (SQLite)

**What Happens:**

```python
# Line 847-867 in main.py
history_records = [
    {
        "memory_id": "a1b2c3d4-...",
        "old_memory": None,
        "new_memory": "User traveled to Japan last month",
        "event": "ADD",
        "created_at": "2025-05-02T19:45:40.371+05:00",
        "is_deleted": 0
    },
    # ... more records
]

self.db.batch_add_history(history_records)

# In storage.py:_create_history_table()
INSERT INTO history (memory_id, old_memory, new_memory, event, created_at, is_deleted)
VALUES (
    "a1b2c3d4-...",
    NULL,
    "User traveled to Japan last month",
    "ADD",
    "2025-05-02T19:45:40.371+05:00",
    0
)
```

**Audit Trail:**
```sql
SELECT * FROM history WHERE memory_id = "a1b2c3d4-..." ORDER BY created_at;

-- Result:
-- memory_id | old_memory | new_memory | event | created_at | is_deleted
-- a1b2c3d4  | NULL       | User trav… | ADD   | 2025-05-02 | 0
-- a1b2c3d4  | User trav… | User now… | UPDATE | 2025-05-03 | 0
-- a1b2c3d4  | User now… | NULL      | DELETE | 2025-05-04 | 1
```

---

## 3. SEARCH WORKFLOW: HOW MEMORIES ARE QUERIED

### A. SEARCH INITIATION (User/Client Side)

```python
# mem0/memory/main.py:1130-1241
memory.search(
    query="Where did I travel recently?",
    filters={"user_id": "user123", "agent_id": "agent_ai"},
    top_k=5,
    threshold=0.1,
    rerank=False  # Optional reranking with cross-encoder
)
```

---

### B. STEP 1: QUERY PREPROCESSING (Client Side)

```python
# Line 1374-1379 in _search_vector_store()

# Lemmatize for BM25
query_lemmatized = lemmatize_for_bm25("Where did I travel recently?")
# → "where travel recent"

# Extract entities from query
query_entities = extract_entities("Where did I travel recently?")
# → [(VERB, "travel"), (ADV, "recently")]

# Create semantic embedding
embeddings = self.embedding_model.embed(query, "search")
# → [0.22, 0.15, ..., 0.88]
```

---

### C. STEP 2-3: DUAL SEARCH (Semantic + Keyword)

```python
# Line 1381-1390 in _search_vector_store()

# SEMANTIC SEARCH - Over-fetch for scoring pool
internal_limit = max(top_k * 4, 60)  # top_k=5 → fetch 60
semantic_results = self.vector_store.search(
    query="Where did I travel recently?",
    vectors=embeddings,  # [0.22, 0.15, ..., 0.88]
    top_k=internal_limit,  # 60
    filters={"user_id": "user123", "agent_id": "agent_ai"}
)
# Returns: [
#   {id: "a1b2...", score: 0.92, payload: {...}},  # "User traveled to Japan"
#   {id: "b2c3...", score: 0.88, payload: {...}},  # "User vacationed in Paris"
#   {id: "c3d4...", score: 0.78, payload: {...}},  # "User likes traveling"
#   ... 57 more results
# ]

# KEYWORD SEARCH - BM25 (Lucene-style)
keyword_results = self.vector_store.keyword_search(
    query=query_lemmatized,  # "where travel recent"
    top_k=internal_limit,    # 60
    filters={"user_id": "user123", "agent_id": "agent_ai"}
)
# Returns: [
#   {id: "a1b2...", score: 8.5},  # "User traveled to Japan" - high BM25
#   {id: "d4e5...", score: 6.2},  # "Travel document for visa"
#   ... 58 more results
# ]
```

**Why Two Searches?**
- **Semantic (Dense Vector)**: Finds semantically similar memories even if text doesn't match
- **Keyword (BM25 Sparse)**: Finds exact term matches, handles rare words better
- **Combined**: Best of both worlds (hybrid search)

---

### D. STEP 4-5: COMPUTE BM25 SCORES

```python
# Line 1392-1400 in _search_vector_store()

bm25_scores = {}
if keyword_results:
    # Parametrize BM25 based on query
    midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
    # Converts raw BM25 to 0-1 range for fusion
    
    for mem in keyword_results:
        raw_score = mem.score  # e.g., 8.5
        if raw_score > 0:
            bm25_scores[mem.id] = normalize_bm25(raw_score, midpoint, steepness)
            # bm25_scores["d4e5..."] = 0.82

bm25_scores = {
    "a1b2...": 0.91,  # Excellent BM25 match
    "d4e5...": 0.82,  # Good BM25 match
    "e5f6...": 0.65   # Moderate BM25 match
}
```

---

### E. STEP 6: ENTITY BOOST COMPUTATION

```python
# Line 1402-1405 in _search_vector_store()

query_entities = [("VERB", "travel"), ("ADV", "recently")]
entity_boosts = self._compute_entity_boosts(query_entities, filters)

# In _compute_entity_boosts() [Line 1466-1525]:
# For each entity in query:
entity_boosts = {}
for entity_text in query_entities:
    entity_embedding = self.embedding_model.embed(entity_text, "search")
    
    # Search entity store
    matches = self.entity_store.search(
        query=entity_text,
        vectors=entity_embedding,
        top_k=500,
        filters={"user_id": "user123", "agent_id": "agent_ai"},
        threshold=0.5  # High similarity required
    )
    
    # For each matched entity, boost its linked memories
    for match in matches:
        similarity = match.score  # e.g., 0.87 (87% similar to "travel")
        if similarity < 0.5:
            continue
        
        linked_memory_ids = match.payload.get("linked_memory_ids", [])
        # e.g., ["a1b2...", "b2c3...", "c3d4..."]
        
        # Spread-attenuated boost (entities linked to many memories get less boost)
        num_linked = max(len(linked_memory_ids), 1)
        memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
        
        boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
        # boost = 0.87 * 0.5 * 0.95 ≈ 0.413
        
        for memory_id in linked_memory_ids:
            entity_boosts[memory_id] = max(entity_boosts.get(memory_id, 0.0), boost)

entity_boosts = {
    "a1b2...": 0.41,  # Linked to "travel" entity
    "b2c3...": 0.38,  # Also linked to "travel"
    "c3d4...": 0.10   # Weakly linked
}
```

**Why Entity Boosts?**
- Memories explicitly mentioning "travel" get boosted
- Prevents semantic drift (find "tourism" even if not semantically close)

---

### F. STEP 7-8: SCORE AND RANK

```python
# Line 1417-1424 in _search_vector_store()

scored_results = score_and_rank(
    semantic_results=candidates,      # 60 results with semantic scores
    bm25_scores=bm25_scores,          # BM25 normalized scores
    entity_boosts=entity_boosts,      # Entity linking boosts
    threshold=threshold,               # 0.1 (min score to include)
    top_k=limit                       # 5 (final result count)
)

# Final Scoring Formula (in scoring.py:score_and_rank):
# final_score = 0.4 * semantic + 0.35 * bm25 + entity_boost
# final_score = 0.4 * 0.92 + 0.35 * 0.91 + 0.41
#             = 0.368 + 0.3185 + 0.41 = 1.0965 (clipped to 1.0)

scored_results = [
    {id: "a1b2...", score: 0.95, payload: {...}},  # Top result
    {id: "b2c3...", score: 0.87, payload: {...}},  # 2nd
    {id: "c3d4...", score: 0.72, payload: {...}},  # 3rd
    {id: "e5f6...", score: 0.61, payload: {...}},  # 4th
    {id: "f6g7...", score: 0.55, payload: {...}}   # 5th
]
```

---

### G. STEP 9: FORMAT AND RETURN

```python
# Line 1426-1464 in _search_vector_store()

original_memories = []
for scored in scored_results:
    payload = scored.get("payload") or {}
    
    memory_item = {
        "id": "a1b2...",
        "memory": "User traveled to Japan last month",
        "hash": "abc123...",
        "created_at": "2025-05-02T19:45:40.371+05:00",
        "updated_at": "2025-05-02T19:45:40.371+05:00",
        "score": 0.95,
        "user_id": "user123",
        "agent_id": "agent_ai"
    }
    
    # Add extra metadata if present
    if additional_metadata:
        memory_item["metadata"] = additional_metadata
    
    original_memories.append(memory_item)

return {
    "results": [
        {
            "id": "a1b2...",
            "memory": "User traveled to Japan last month",
            "score": 0.95,
            ...
        },
        # ... 4 more
    ]
}
```

---

## 4. MEMORY INTEGRATION INTO AI LLM PROMPTS

### Key Pattern: HOW THE AI AGENT USES MEMORIES

**Scenario**: User asks agent for advice; agent needs to be "aware" of user's memories.

```python
# Client-side code (not shown in API, but typical pattern)

# Step 1: Search for relevant memories
memories = memory.search(
    query=user_input,
    filters={"user_id": "user123"}
)

# Step 2: Format memories for LLM prompt
memory_context = format_memories_for_prompt(memories)

# Step 3: Call LLM with memory context
response = llm.generate_response(
    messages=[
        {
            "role": "system",
            "content": """You are a helpful assistant. Use the provided user memories \
to personalize your responses."""
        },
        {
            "role": "user",
            "content": f"""USER MEMORIES:
{memory_context}

USER QUERY: {user_input}"""
        }
    ]
)
```

### Memory Context Format (In Prompt):

```
USER MEMORIES:
- User traveled to Japan last month (Score: 0.95)
- User prefers Python programming (Score: 0.92)
- User works as a Software Engineer (Score: 0.88)

USER QUERY: What programming language should I learn next?
```

---

## 5. UPDATE & DELETE WORKFLOWS

### A. UPDATE MEMORY

```python
# mem0/memory/main.py:1527-1548

memory.update(
    memory_id="a1b2c3d4-...",
    data="User traveled to Japan and loved the culture",
    metadata={"experience": "enriched"}
)
```

**Process:**

```python
# Line 1683-1746 in _update_memory()

# 1. Fetch existing memory
existing_memory = self.vector_store.get(vector_id=memory_id)
prev_value = existing_memory.payload.get("data")
# → "User traveled to Japan last month"

# 2. Create new embedding
new_embedding = self.embedding_model.embed(new_data, "update")

# 3. Update vector store
self.vector_store.update(
    vector_id=memory_id,
    vector=new_embedding,
    payload={
        "data": "User traveled to Japan and loved the culture",
        "hash": hashlib.md5(new_data.encode()).hexdigest(),  # New hash
        "text_lemmatized": lemmatize_for_bm25(new_data),
        "created_at": existing_memory.payload.get("created_at"),  # Preserve
        "updated_at": datetime.now(timezone.utc).isoformat(),  # New timestamp
        ...existing metadata...
    }
)

# 4. Record in history
self.db.add_history(
    memory_id=memory_id,
    old_memory=prev_value,
    new_memory=new_data,
    event="UPDATE",
    ...
)
# Adds row: UPDATE event with both old and new values

# 5. Clean up and re-link entities
self._remove_memory_from_entity_store(memory_id, filters)
# Remove memory_id from all entity.linked_memory_ids

self._link_entities_for_memory(memory_id, new_data, filters)
# Extract new entities from new_data and link them
```

---

### B. DELETE MEMORY

```python
# mem0/memory/main.py:1550-1564

memory.delete(memory_id="a1b2c3d4-...")
```

**Process:**

```python
# Line 1748-1776 in _delete_memory()

# 1. Fetch memory before deletion
existing_memory = self.vector_store.get(vector_id=memory_id)
prev_value = existing_memory.payload.get("data")

# 2. Delete from vector store
self.vector_store.delete(vector_id=memory_id)

# 3. Record in history (soft delete marker)
self.db.add_history(
    memory_id=memory_id,
    old_memory=prev_value,
    new_memory=None,  # NULL indicates deletion
    event="DELETE",
    is_deleted=1  # Soft delete flag
)

# 4. Remove from entity store
self._remove_memory_from_entity_store(memory_id, filters)
# Strips memory_id from all entity.linked_memory_ids
```

---

## 6. PARALLEL OPERATIONS SUMMARY

### What Runs in Parallel (Within Single `add()` Call):

| Operation | Phase | Parallelization |
|-----------|-------|-----------------|
| Last messages fetch + Vector search | 1 | Independent queries (can run simultaneously) |
| Batch embed texts | 3 | Single call to embedding API (vectors computed in parallel) |
| Batch embed entities | 7b | Single call (all entity embeddings in parallel) |
| Batch entity search | 7c | Single search_batch call (all entity searches in parallel) |
| Batch entity insert/update | 7d-e | Grouped inserts/updates in parallel |

### What is Sequential:

| Operation | Reason |
|-----------|--------|
| LLM extraction (Phase 2) | Waits for Phase 1 (existing memories needed) |
| Vector store insert (Phase 6) | Waits for Phase 3 (embeddings needed) |
| Entity extraction (Phase 7) | Waits for Phase 6 (memory_ids needed for linking) |
| History record (Phase 9) | Waits for Phase 6 (needs vector IDs) |

---

## 7. CLIENT VS SERVER RESPONSIBILITIES

### User/Client Side (Where SDK Runs):

```
[User Device / Client Code]
├── Validation & Parsing (Phase 0)
├── Database queries (Phase 1 - last messages)
├── Embedding generation (Phases 3, 7b, 7d)
├── Vector store operations (Phases 1, 6, 7)
├── Hash deduplication (Phase 5)
├── Entity extraction & linking (Phase 7)
├── SQLite history tracking (Phase 9)
└── Search ranking & scoring (All search phases)
```

### AI LLM Side (Happens Once Per Add):

```
[LLM Service]
└── Memory Extraction (Phase 2)
    ├── Receives existing memories + new messages
    ├── Returns NEW memories to add
    └── Single JSON response
```

### No Continuous AI Processing:
- The AI is NOT continuously monitoring or updating memories
- Only called ONCE per `add()` to extract facts
- All storage, deduplication, and ranking is CLIENT-SIDE

---

## 8. MEMORY STORAGE SCHEMA

### Vector Store (Qdrant/Pinecone/etc):

```json
{
  "id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
  "vector": [0.15, 0.22, ..., 0.89],
  "payload": {
    "data": "User traveled to Japan last month",
    "hash": "abc123def456...",
    "created_at": "2025-05-02T19:45:40.371+05:00",
    "updated_at": "2025-05-02T19:45:40.371+05:00",
    "text_lemmatized": "user travel japan month",
    "attributed_to": "user",
    "role": "user",
    "user_id": "user123",
    "agent_id": "agent_ai",
    "run_id": null,
    "actor_id": null,
    "metadata": {
      "custom_field": "custom_value"
    }
  }
}
```

### Entity Store (Separate Collection):

```json
{
  "id": "entity_uuid_...",
  "vector": [0.12, 0.34, ..., 0.78],
  "payload": {
    "data": "Japan",
    "entity_type": "LOCATION",
    "linked_memory_ids": [
      "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
      "b2c3d4e5-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
      "c3d4e5f6-e5f6-g7h8-i9j0-k1l2m3n4o5p6"
    ],
    "user_id": "user123",
    "agent_id": "agent_ai"
  }
}
```

### SQLite Messages Table (Short-term):

```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    content TEXT,
    role TEXT,  -- "user", "assistant", "system"
    session_scope TEXT,  -- "user_id=X&agent_id=Y"
    created_at DATETIME
);

-- Example rows:
-- "msg_001" | "I like Python" | "user" | "user_id=user123" | "2025-05-02 19:45:40"
-- "msg_002" | "Great choice!" | "assistant" | "user_id=user123" | "2025-05-02 19:45:45"
```

### SQLite History Table (Audit):

```sql
CREATE TABLE history (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    old_memory TEXT,
    new_memory TEXT,
    event TEXT,  -- "ADD", "UPDATE", "DELETE", "NONE"
    created_at DATETIME,
    updated_at DATETIME,
    is_deleted INTEGER,  -- 0 or 1
    actor_id TEXT,
    role TEXT
);

-- Example rows:
-- "hist_001" | "a1b2..." | NULL | "User traveled to Japan" | "ADD" | "2025-05-02" | "2025-05-02" | 0 | NULL | "user"
-- "hist_002" | "a1b2..." | "User traveled to Japan" | "User traveled to Japan and loved it" | "UPDATE" | "2025-05-02" | "2025-05-03" | 0 | NULL | "user"
-- "hist_003" | "a1b2..." | "User traveled to Japan and loved it" | NULL | "DELETE" | "2025-05-02" | "2025-05-04" | 1 | NULL | "user"
```

---

## 9. THREE MEMORY TYPES IN ACTION

### Type 1: SEMANTIC_MEMORY

```python
# User says: "I prefer Python over Java"
# Extraction:
memory.add(
    messages=[{"role": "user", "content": "I prefer Python over Java"}],
    user_id="user123"
    # memory_type not specified → default to SEMANTIC
)

# Storage:
# Vector Store: {"data": "User prefers Python over Java", "attributed_to": "user", ...}
# Note: NO "memory_type" field explicitly stored; it's implicit in vector store
```

---

### Type 2: EPISODIC_MEMORY

```python
# User says: "I attended a Python conference in SF last week"
# Extraction:
memory.add(
    messages=[{"role": "user", "content": "I attended a Python conference in SF last week"}],
    user_id="user123"
    # memory_type not specified → default to SEMANTIC
)

# Storage:
# Vector Store: {"data": "User attended a Python conference in SF last week", "attributed_to": "user", ...}
# Note: In current implementation, episodic memories are stored same as semantic
#       Distinction is often made at SEARCH time (e.g., temporal queries) not storage
```

---

### Type 3: PROCEDURAL_MEMORY (Agent Only)

```python
# Agent says: "I need to summarize my execution steps"
# Extraction:
memory.add(
    messages=[
        {"role": "user", "content": "I visited 5 websites"},
        {"role": "assistant", "content": "I extracted data from each..."}
    ],
    agent_id="agent_ai",  # REQUIRED for procedural
    memory_type="procedural_memory"  # EXPLICIT flag
)

# Process:
# 1. All messages → LLM with PROCEDURAL_MEMORY_SYSTEM_PROMPT
# 2. LLM creates summary of agent execution history
# 3. Summary stored in vector store with memory_type="procedural_memory"

# Storage:
# Vector Store: {
#    "data": "Agent visited 5 websites. Site 1: extracted product list (25 items)...",
#    "memory_type": "procedural_memory",
#    "agent_id": "agent_ai",
#    "attributed_to": "assistant"
# }
```

---

## 10. COMPLETE FLOW DIAGRAM (Text)

```
USER INPUT
    ↓
[CLIENT VALIDATION & PARSING]
    ├─→ Normalize IDs
    ├─→ Convert to message list
    └─→ Build session scope
         ↓
[PARALLEL: DB + VECTOR SEARCH]
    ├─→ Get last 10 messages (SQLite)
    └─→ Search existing memories (Vector Store)
         ↓
[LLM EXTRACTION - AI CALL #1]
    ├─→ Send: existing memories + new messages + context
    └─→ Receive: JSON with new memories to add
         ↓
[BATCH EMBED NEW MEMORIES]
    └─→ embed_batch(all_texts) → get vectors
         ↓
[HASH DEDUP + LEMMATIZE]
    ├─→ Skip duplicates (MD5 hash)
    └─→ Lemmatize for BM25
         ↓
[BATCH INSERT TO VECTOR STORE]
    └─→ All vectors + payloads in single call
         ↓
[BATCH ENTITY EXTRACTION & LINKING]
    ├─→ extract_entities_batch(all_texts)
    ├─→ Deduplicate entities globally
    ├─→ embed_batch(entity_texts)
    ├─→ search_batch(existing entities)
    └─→ Batch insert/update entities
         ↓
[PERSIST SHORT-TERM MESSAGES]
    └─→ SQLite: Save raw messages
         ↓
[RECORD HISTORY]
    └─→ SQLite: ADD event for each memory
         ↓
RETURN: {"results": [added memory IDs and texts]}


========== SEARCH WORKFLOW ==========

USER SEARCH QUERY
    ↓
[PREPROCESS]
    ├─→ Lemmatize query
    └─→ Extract entities from query
         ↓
[DUAL SEARCH]
    ├─→ Semantic: embed(query) → vector search (top 60)
    └─→ Keyword: keyword_search(lemmatized) (top 60)
         ↓
[COMPUTE SCORES]
    ├─→ Normalize BM25 scores to 0-1
    ├─→ Compute entity boosts (from entity store search)
    └─→ Combine: 0.4*semantic + 0.35*bm25 + entity_boost
         ↓
[RANK & FILTER]
    ├─→ Keep only scores ≥ threshold (0.1)
    └─→ Return top K (5)
         ↓
[FORMAT & RETURN]
    └─→ {"results": [top 5 memories with scores]}


========== UPDATE WORKFLOW ==========

UPDATE REQUEST (memory_id, new_text)
    ↓
[FETCH EXISTING]
    └─→ Get from vector store
         ↓
[RE-EMBED NEW TEXT]
    └─→ embedding(new_text) → new vector
         ↓
[UPDATE VECTOR STORE]
    └─→ Overwrite vector + payload + timestamp
         ↓
[RECORD HISTORY]
    └─→ SQLite: UPDATE event (old_memory + new_memory)
         ↓
[ENTITY CLEANUP]
    ├─→ Remove memory_id from old entity links
    └─→ Extract & link entities from new text
         ↓
RETURN: {"message": "Memory updated successfully!"}


========== DELETE WORKFLOW ==========

DELETE REQUEST (memory_id)
    ↓
[FETCH & DELETE]
    ├─→ Get from vector store (before deletion)
    └─→ Delete from vector store
         ↓
[RECORD HISTORY]
    └─→ SQLite: DELETE event (is_deleted=1, new_memory=NULL)
         ↓
[ENTITY CLEANUP]
    └─→ Remove memory_id from all entity links
         ↓
RETURN: {"message": "Memory deleted successfully!"}
```

---

## 11. CRITICAL INSIGHTS

### 1. **No Real-Time Agent Monitoring**
- Agent is NOT continuously updated about user memories
- Only memories at the time of `add()` call are extracted
- Agent must explicitly `search()` or the app developer must inject memories into prompts

### 2. **Hybrid Search Architecture**
- Semantic vectors handle meaning
- BM25 keywords handle exact terms
- Entity boosts add knowledge graph layer
- **Result**: Better recall & precision than any single method

### 3. **Entity Store is Separate**
- Not just "metadata" on memories
- Full vector store dedicated to entities
- Enables fast entity-based queries and boosts
- **Example**: "All memories about Japan" → search entity store first

### 4. **Asynchronous Processing NOT Built-In**
- SDK provides `AsyncMemory` class
- But internally, uses `asyncio.to_thread()` for I/O
- All logic still sequential (not truly async at business logic level)

### 5. **Three Memory Layers**
1. **Vector Store** (Long-term): Semantic + Episodic + Procedural (all types)
2. **Entity Store** (Knowledge Graph): Entity-to-memories mapping
3. **SQLite** (Short-term): Last K messages + History audit trail

### 6. **Soft Deletes in History**
- Memories are deleted from vector store but kept in history
- `history()` query shows all past versions
- Enables "undo" functionality if needed

### 7. **Session Scoping**
- Each operation scoped to user_id / agent_id / run_id
- Prevents memory leakage between users/agents
- Built into all vector store searches via filters

---

## 12. PERFORMANCE CHARACTERISTICS

| Operation | Time | Notes |
|-----------|------|-------|
| Add (N memories) | ~2-5s | Dominated by LLM extraction (Phase 2) |
| Search | ~500ms | Parallel semantic + keyword + entity boost |
| Update | ~1s | Re-embed + entity relinking |
| Delete | ~500ms | Vector delete + entity cleanup |
| Get (single) | ~50ms | Direct vector store lookup |
| Get All | ~200ms | List with filters |

**Bottlenecks:**
1. LLM extraction (Phase 2) - 70-80% of add() time
2. Embedding generation (Phases 3, 7b) - 10-15% of add() time
3. Vector store I/O - 5-10% of add() time

---

## 13. EXAMPLE: END-TO-END FLOW

```
[USER]
"I just got promoted to Senior Engineer at Shopify!"

↓

[PHASE 0-1: Parsing + Search Existing]
- Parsed: "User promoted to Senior Engineer at Shopify"
- Found existing: "User works at Shopify" (score 0.88)

↓

[PHASE 2: LLM Extraction]
System: "Extract relevant facts, avoiding duplicates of existing memories"
User: "New message: I just got promoted to Senior Engineer at Shopify!
       Existing memories: [User works at Shopify]"

LLM Returns:
{
  "memory": [
    {
      "id": "0",
      "text": "User was promoted to Senior Engineer at Shopify",
      "attributed_to": "user",
      "linked_memory_ids": ["existing_uuid_123"]
    }
  ]
}

↓

[PHASE 3: Batch Embed]
embed("User was promoted to Senior Engineer at Shopify")
→ [0.18, 0.22, ..., 0.91]

↓

[PHASES 4-6: Deduplicate + Insert]
- Hash: "abc789def012..."
- Metadata: {data, hash, created_at, user_id, agent_id, ...}
- INSERT to vector store ✓

↓

[PHASE 7: Entity Extraction & Linking]
extract_entities("User was promoted to Senior Engineer at Shopify")
→ [("PERSON_TITLE", "Senior Engineer"), ("ORG", "Shopify")]

- "Senior Engineer" entity: search_existing → update (link to new memory)
- "Shopify" entity: search_existing → update (link to new memory)

✓ Entity store updated with memory link

↓

[PHASE 8-9: Persist]
- SQLite messages: save raw message
- SQLite history: record ADD event

↓

RETURN:
{
  "results": [
    {
      "id": "new_uuid_456",
      "memory": "User was promoted to Senior Engineer at Shopify",
      "event": "ADD"
    }
  ]
}

---

[LATER: USER ASKS FOR ADVICE]

query = "I want to negotiate a new salary based on my career growth"

↓

[Search: Preprocess]
lemmatize: "want negotiate salary career growth"
entities: [("NOUN", "salary"), ("NOUN", "career")]

↓

[Search: Dual Search]
semantic_results (60):
- "User promoted to Senior Engineer at Shopify" (0.89)
- "User negotiated contracts before" (0.76)
- ...

keyword_results (60):
- "User promoted to Senior Engineer" (BM25: 7.2)
- "Career goals: reach management" (BM25: 5.8)
- ...

↓

[Search: Entity Boost]
Entity "Shopify": links to ["promoted_memory", "company_memory", ...]
Entity "salary": links to ["negotiation_memory", ...]
→ Boost these memories

↓

[Search: Score & Rank]
final_score = 0.4 * semantic + 0.35 * bm25 + entity_boost

Top 3 results:
1. "User promoted to Senior Engineer at Shopify" (0.94)
2. "User negotiated contracts before" (0.82)
3. "Career goals: reach management" (0.71)

↓

[Format & Return]
{
  "results": [
    {
      "id": "new_uuid_456",
      "memory": "User was promoted to Senior Engineer at Shopify",
      "score": 0.94
    },
    ...
  ]
}

↓

[AI APP INJECTS INTO PROMPT]
System: "Use this context about the user to provide personalized advice"
User: "MEMORIES:
       - User was promoted to Senior Engineer at Shopify (Score: 0.94)
       - User negotiated contracts before (Score: 0.82)
       
       QUERY: I want to negotiate a new salary...
       
       LLM thinks: Ah, this user just got promoted AND has negotiation experience.
       I should recommend leveraging both points in the salary negotiation."
```

---

## SUMMARY

Mem0's architecture is a **sophisticated hybrid system**:

1. **On User/Client Side**: Heavy computational lifting
   - Deduplication, embeddings, entity extraction, ranking
   - SQLite for audit trail & short-term context
   - Vector + Entity stores for long-term knowledge

2. **On AI LLM Side**: Single-shot extraction
   - Sees existing memories + new messages
   - Returns NEW facts to add
   - No continuous monitoring

3. **Hybrid Search**: Best of three worlds
   - Semantic (vectors) for meaning
   - Keyword (BM25) for exact terms
   - Entities (knowledge graph) for relationships

4. **Asynchronous Composition**: 
   - Batch operations where possible (embedding, entity extraction)
   - Sequential dependencies where necessary (LLM→vectors→entities)

5. **Safety & Privacy**:
   - Session-scoped (user_id, agent_id, run_id filters)
   - Audit trail (history table tracks all changes)
   - Soft deletes (nothing truly erased, marked as deleted)

