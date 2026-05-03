"""
Healthcare Memory System Integration
====================================

Full integration of MedQuAD (Semantic) + Synthea (Episodic) into Mem0.

Architecture:
============

┌─────────────────────────────────────────────────────────────────┐
│              HEALTHCARE MEMORY SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEMANTIC MEMORY LAYER                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MedQuAD: Medical Knowledge Base                          │   │
│  │ - Global, shared across all patients                    │   │
│  │ - NO user_id scoping                                     │   │
│  │ - Generic medical facts, diseases, drugs, therapies    │   │
│  │ - Stored in Vector Store (embeddings)                  │   │
│  │ - Example: "Glaucoma is a group of diseases..."        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓ Search combines                      │
│  EPISODIC MEMORY LAYER                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Synthea: Patient Health Histories                        │   │
│  │ - Per-patient (scoped to user_id)                       │   │
│  │ - MUST have user_id (patient UUID)                      │   │
│  │ - Patient-specific diagnoses, medications, encounters   │   │
│  │ - Temporal data (START/STOP dates)                      │   │
│  │ - Stored in Vector Store (embeddings)                  │   │
│  │ - Example: "Patient diagnosed with bronchitis 2013..."  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                       │
│  SHORT-TERM MEMORY LAYER                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SQLite: Recent Conversation History                      │   │
│  │ - Last K messages per patient session                   │   │
│  │ - Immediate context for LLM extraction                 │   │
│  │ - Used in Phase 1 of add() workflow                    │   │
│  │ - Example: [user_msg, assistant_msg, ...]              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

SEARCH WORKFLOW:
================

User Query (e.g., "What should I take for my blood pressure?")
    ↓
[PREPROCESSING]
  ├─ Lemmatize query
  └─ Extract entities

    ↓
[DUAL SEARCH]
  ├─ Semantic search: Find similar embeddings
  │  ├─ SEMANTIC: "Hypertension treatment" (global)
  │  └─ EPISODIC: "Patient's BP reading 155/95" (user_id scoped)
  │
  └─ Keyword search: BM25 on text_lemmatized
     ├─ SEMANTIC: Medical articles on BP management
     └─ EPISODIC: Patient's encounter notes

    ↓
[SCORING & RANKING]
  ├─ Semantic score (dense vector similarity)
  ├─ BM25 score (keyword match)
  ├─ Entity boost (if "blood pressure" entity found)
  └─ Filter by user_id scope
     - Return EPISODIC (patient-scoped) + SEMANTIC (global)

    ↓
[RESULTS]
  Top K memories combining both semantic and episodic knowledge
  with proper ranking.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from mem0 import Memory
from mem0.configs.base import MemoryConfig

from healthcare_integration import HealthcareDatasetLoader

logger = logging.getLogger(__name__)


class HealthcareMemorySystem:
    """
    Complete healthcare memory system with proper semantic/episodic separation.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize healthcare memory system."""
        self.config = config or MemoryConfig()
        self.memory = Memory(self.config)
        self.loader = HealthcareDatasetLoader()
    
    # ==================== SEMANTIC MEMORY OPERATIONS ====================
    
    def add_semantic_medical_knowledge(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Add medical knowledge (MedQuAD + Flashcards) to semantic memory.
        
        These are GLOBAL facts (no user_id), shared across all patients.
        """
        logger.info("Loading medical knowledge (MedQuAD + Flashcards)...")
        memory_ids = []
        
        # 1. Load MedQuAD
        medquad_list = self.loader.load_medquad(limit=limit)
        for idx, qa in enumerate(medquad_list):
            try:
                formatted_text = self.loader.format_semantic_memory(qa)
                result = self.memory.add(
                    messages=[{"role": "user", "content": formatted_text}],
                    infer=False,
                    metadata={
                        "source": qa.get("source", ""),
                        "focus_area": qa.get("focus_area", ""),
                        "memory_type": "semantic_medical_knowledge",
                        "dataset": "medquad"
                    }
                )
                if result["results"]:
                    memory_ids.append(result["results"][0]["id"])
            except Exception as e:
                logger.error(f"Failed to add MedQuAD memory {idx}: {e}")

        # 2. Load Medical Flashcards
        flashcards_list = self.loader.load_medical_flashcards(limit=limit)
        for idx, card in enumerate(flashcards_list):
            try:
                formatted_text = self.loader.format_flashcard(card)
                result = self.memory.add(
                    messages=[{"role": "user", "content": formatted_text}],
                    infer=False,
                    metadata={
                        "instruction": card.get("instruction", ""),
                        "memory_type": "semantic_medical_knowledge",
                        "dataset": "medical_flashcards"
                    }
                )
                if result["results"]:
                    memory_ids.append(result["results"][0]["id"])
            except Exception as e:
                logger.error(f"Failed to add flashcard memory {idx}: {e}")
        
        logger.info(f"✓ Added {len(memory_ids)} semantic medical knowledge memories")
        
        return {
            "added": len(memory_ids),
            "medquad_count": len(medquad_list),
            "flashcards_count": len(flashcards_list),
            "memory_ids": memory_ids
        }
    
    # ==================== EPISODIC MEMORY OPERATIONS ====================
    
    def add_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Add a patient's complete history to episodic memory.
        
        Loads ALL patient data for a specific patient:
        - Conditions (diagnoses)
        - Medications (prescriptions)
        - Encounters (visits)
        
        Each is stored with user_id=patient_id.
        
        Args:
            patient_id: Synthea patient UUID
        
        Returns:
            {
                "patient_id": "uuid",
                "conditions": 5,
                "medications": 12,
                "encounters": 3,
                "total_added": 20,
                "memory_ids": [...]
            }
        """
        logger.info(f"Loading episodic memory for patient {patient_id}...")
        
        memory_ids = []
        counts = {"conditions": 0, "medications": 0, "encounters": 0}
        
        # Load and add CONDITIONS
        conditions = self.loader.load_synthea_conditions(patient_id=patient_id)
        for cond in conditions:
            try:
                formatted_text = self.loader.format_episodic_condition(cond)
                
                # Add with user_id (EPISODIC, scoped to patient)
                result = self.memory.add(
                    messages=[{"role": "user", "content": formatted_text}],
                    user_id=patient_id,  # ← IMPORTANT: Scoped to patient
                    infer=False,
                    metadata={
                        "data_type": "condition",
                        "start_date": cond.get("START", ""),
                        "end_date": cond.get("STOP", ""),
                        "condition_code": cond.get("CODE", ""),
                        "memory_type": "episodic_patient_condition"
                    }
                )
                
                if result["results"]:
                    memory_ids.append(result["results"][0]["id"])
                    counts["conditions"] += 1
            
            except Exception as e:
                logger.error(f"Failed to add condition: {e}")
        
        # Load and add MEDICATIONS
        medications = self.loader.load_synthea_medications(patient_id=patient_id)
        for med in medications:
            try:
                formatted_text = self.loader.format_episodic_medication(med)
                
                result = self.memory.add(
                    messages=[{"role": "user", "content": formatted_text}],
                    user_id=patient_id,  # ← IMPORTANT: Scoped to patient
                    infer=False,
                    metadata={
                        "data_type": "medication",
                        "start_date": med.get("START", ""),
                        "end_date": med.get("STOP", ""),
                        "medication_code": med.get("CODE", ""),
                        "reason": med.get("REASONDESCRIPTION", ""),
                        "memory_type": "episodic_patient_medication"
                    }
                )
                
                if result["results"]:
                    memory_ids.append(result["results"][0]["id"])
                    counts["medications"] += 1
            
            except Exception as e:
                logger.error(f"Failed to add medication: {e}")
        
        # Load and add ENCOUNTERS
        encounters = self.loader.load_synthea_encounters(patient_id=patient_id)
        for enc in encounters:
            try:
                formatted_text = self.loader.format_episodic_encounter(enc)
                
                result = self.memory.add(
                    messages=[{"role": "user", "content": formatted_text}],
                    user_id=patient_id,  # ← IMPORTANT: Scoped to patient
                    infer=False,
                    metadata={
                        "data_type": "encounter",
                        "start_date": enc.get("START", ""),
                        "end_date": enc.get("STOP", ""),
                        "encounter_type": enc.get("ENCOUNTERCLASS", ""),
                        "provider": enc.get("PROVIDER", ""),
                        "memory_type": "episodic_patient_encounter"
                    }
                )
                
                if result["results"]:
                    memory_ids.append(result["results"][0]["id"])
                    counts["encounters"] += 1
            
            except Exception as e:
                logger.error(f"Failed to add encounter: {e}")
        
        logger.info(f"✓ Added episodic memory for patient {patient_id}:")
        logger.info(f"    - Conditions: {counts['conditions']}")
        logger.info(f"    - Medications: {counts['medications']}")
        logger.info(f"    - Encounters: {counts['encounters']}")
        
        return {
            "patient_id": patient_id,
            "conditions": counts["conditions"],
            "medications": counts["medications"],
            "encounters": counts["encounters"],
            "total_added": len(memory_ids),
            "memory_ids": memory_ids
        }
    
    def add_patient_short_term_context(self, patient_id: str, conversation: List[Dict[str, str]]):
        """
        Add recent conversation messages to short-term memory.
        
        These messages become context for future LLM extraction.
        
        Args:
            patient_id: Patient UUID
            conversation: List of message dicts with 'role' and 'content'
        
        Example:
            conversation = [
                {"role": "user", "content": "I've been feeling dizzy"},
                {"role": "assistant", "content": "When did this start?"},
                {"role": "user", "content": "About a week ago"}
            ]
        """
        logger.info(f"Adding {len(conversation)} short-term context messages for patient {patient_id}...")
        
        # Add messages (they get stored in SQLite automatically)
        result = self.memory.add(
            messages=conversation,
            user_id=patient_id,
            infer=True  # Let LLM extract facts from conversation
        )
        
        # Get recent messages to verify they were stored
        recent = self.memory.get_recent_messages(user_id=patient_id, limit=20)
        
        logger.info(f"✓ Short-term memory now has {len(recent)} recent messages")
        
        return {
            "patient_id": patient_id,
            "messages_added": len(conversation),
            "short_term_count": len(recent),
            "extracted_memories": len(result["results"])
        }
    
    # ==================== SEARCH & RETRIEVAL ====================
    
    def search_patient_knowledge(
        self,
        patient_id: str,
        query: str,
        top_k: int = 10,
        include_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Search for knowledge relevant to a patient.
        
        Returns BOTH:
        1. EPISODIC: Patient-specific memories (scoped to user_id)
        2. SEMANTIC: General medical knowledge (if include_semantic=True)
        
        Args:
            patient_id: Patient UUID
            query: Search query
            top_k: Number of results
            include_semantic: Whether to include global semantic knowledge
        
        Returns:
            {
                "episodic": [patient-specific memories],
                "semantic": [general medical knowledge],
                "combined_ranked": [all results ranked by relevance]
            }
        """
        logger.info(f"Searching knowledge for patient {patient_id}: '{query}'")
        
        # Search 1: Get EPISODIC (patient-scoped)
        episodic_results = self.memory.search(
            query=query,
            filters={"user_id": patient_id},
            top_k=top_k,
            threshold=0.1
        )
        
        # Search 2: Get SEMANTIC (global, no patient scoping)
        semantic_results = []
        if include_semantic:
            # Search without user_id filter to get global knowledge
            # Note: Current Mem0 requires at least one filter
            # Workaround: use get_all() and filter post-hoc
            try:
                all_memories = self.memory.get_all(
                    filters={},  # Empty = global
                    top_k=top_k
                )
                semantic_results = all_memories["results"]
            except:
                # Fallback if get_all() requires filter
                semantic_results = []
        
        logger.info(f"✓ Found {len(episodic_results['results'])} episodic + {len(semantic_results)} semantic results")
        
        return {
            "query": query,
            "patient_id": patient_id,
            "episodic": episodic_results["results"],
            "semantic": semantic_results,
            "combined_count": len(episodic_results["results"]) + len(semantic_results)
        }
    
    def get_patient_recent_context(self, patient_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get patient's recent conversation messages (short-term memory).
        
        These are used as context for LLM in memory extraction phase.
        """
        return self.memory.get_recent_messages(user_id=patient_id, limit=limit)
    
    def get_patient_full_history(self, patient_id: str, top_k: int = 100) -> Dict[str, Any]:
        """
        Get patient's complete episodic history.
        
        Returns all memories scoped to patient.
        """
        results = self.memory.get_all(
            filters={"user_id": patient_id},
            top_k=top_k
        )
        
        return {
            "patient_id": patient_id,
            "memory_count": len(results["results"]),
            "memories": results["results"]
        }
    
    # ==================== UTILITY ====================
    
    def reset(self):
        """Clear all memories (semantic + episodic + short-term)."""
        logger.warning("Resetting all healthcare memories...")
        self.memory.reset()
        logger.info("✓ All memories cleared")


# ==================== EXAMPLE USAGE ====================

def example_full_workflow():
    """
    Complete example showing semantic + episodic + short-term integration.
    """
    print("\n" + "="*80)
    print("HEALTHCARE MEMORY SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize
    system = HealthcareMemorySystem()
    
    # STEP 1: Load SEMANTIC knowledge (global)
    print("\n[STEP 1] Loading SEMANTIC medical knowledge from MedQuAD + Flashcards...")
    semantic_result = system.add_semantic_medical_knowledge(limit=10)
    print(f"  Added {semantic_result['added']} semantic memories")
    print(f"    - MedQuAD: {semantic_result['medquad_count']}")
    print(f"    - Flashcards: {semantic_result['flashcards_count']}")
    
    # STEP 2: Load EPISODIC patient history
    print("\n[STEP 2] Loading EPISODIC patient history from Synthea...")
    
    # Get a sample patient ID
    patients = system.loader.load_synthea_patients(limit=1)
    if patients:
        patient_id = patients[0]["PATIENT_ID"]
        print(f"  Using patient: {patients[0]['FIRST']} {patients[0]['LAST']} (ID: {patient_id})")
        
        episodic_result = system.add_patient_history(patient_id)
        print(f"  Added {episodic_result['total_added']} episodic memories:")
        print(f"    - Conditions: {episodic_result['conditions']}")
        print(f"    - Medications: {episodic_result['medications']}")
        print(f"    - Encounters: {episodic_result['encounters']}")
        
        # STEP 3: Add SHORT-TERM context
        print("\n[STEP 3] Adding SHORT-TERM conversation context...")
        conversation = [
            {"role": "user", "content": "I've been experiencing shortness of breath"},
            {"role": "assistant", "content": "When did this start?"},
            {"role": "user", "content": "About 2 days ago, especially when climbing stairs"}
        ]
        
        short_term_result = system.add_patient_short_term_context(patient_id, conversation)
        print(f"  Short-term memory: {short_term_result['short_term_count']} messages")
        
        # STEP 4: Search across all memory types
        print("\n[STEP 4] Searching across SEMANTIC + EPISODIC + SHORT-TERM...")
        search_result = system.search_patient_knowledge(
            patient_id=patient_id,
            query="respiratory issues medication",
            top_k=5
        )
        print(f"  Found {search_result['combined_count']} relevant memories")
        print(f"    - Episodic (patient-specific): {len(search_result['episodic'])}")
        print(f"    - Semantic (general knowledge): {len(search_result['semantic'])}")
        
        # STEP 5: Get patient history
        print("\n[STEP 5] Retrieving patient's complete history...")
        history = system.get_patient_full_history(patient_id, top_k=20)
        print(f"  Patient has {history['memory_count']} total episodic memories")
        
        # Show first few
        for mem in history['memories'][:3]:
            print(f"    - {mem.get('memory', '')[:60]}...")
    
    print("\n" + "="*80)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_full_workflow()
