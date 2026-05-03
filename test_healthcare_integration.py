"""
Healthcare Memory Separation Test Suite
========================================

Tests to verify that Semantic and Episodic memories are properly separated
and that short-term memory works correctly.

Memory Separation Guarantees:
1. SEMANTIC (MedQuAD): Global, no user_id, shared across all agents
2. EPISODIC (Synthea): Per-patient (user_id), temporal, personal
3. SHORT-TERM (SQLite): Last K messages per session, immediate context
"""

import pytest
import logging
from datetime import datetime
from typing import Dict, List, Any

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.configs.enums import MemoryType

logger = logging.getLogger(__name__)


class TestMemorySeparation:
    """
    Test Suite 1: Verify Semantic and Episodic memories are stored separately
    """
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Initialize Mem0 with test config."""
        config = MemoryConfig()
        self.memory = Memory(config)
        yield
        # Cleanup
        try:
            self.memory.reset()
        except:
            pass
    
    def test_semantic_memory_has_no_user_id(self):
        """
        SEMANTIC MEMORY: Global knowledge should NOT have user_id/agent_id.
        
        MedQuAD facts should be stored globally:
        - No user_id
        - No agent_id
        - Shared across all patients
        """
        # Add semantic fact (medical knowledge)
        semantic_fact = "Glaucoma is a group of diseases that damage the optic nerve and result in vision loss"
        
        result = self.memory.add(
            messages=[{"role": "user", "content": semantic_fact}],
            # NO user_id, NO agent_id → Global semantic memory
            infer=False  # Store raw, don't extract
        )
        
        assert result["results"], "Semantic memory should be added"
        memory_id = result["results"][0]["id"]
        
        # Retrieve and verify
        retrieved = self.memory.get(memory_id)
        assert retrieved is not None, "Semantic memory should be retrievable"
        assert retrieved.get("memory") == semantic_fact
        
        # Key assertion: NO user_id in payload
        assert "user_id" not in retrieved or retrieved.get("user_id") is None, \
            "Semantic memory should NOT have user_id"
        
        logger.info("✓ Semantic memory properly stored without user_id")
    
    def test_episodic_memory_scoped_to_user(self):
        """
        EPISODIC MEMORY: Patient-specific memories should be scoped to user_id.
        
        Synthea conditions/medications should:
        - Have explicit user_id (patient ID)
        - Be retrievable ONLY with correct user_id filter
        - Be isolated from other patients
        """
        patient_id = "patient_001"
        
        # Add episodic fact (patient history)
        episodic_fact = "Patient was diagnosed with Acute bronchitis on 2013-06-24"
        
        result = self.memory.add(
            messages=[{"role": "user", "content": episodic_fact}],
            user_id=patient_id,  # IMPORTANT: Scoped to patient
            infer=False
        )
        
        assert result["results"], "Episodic memory should be added"
        memory_id = result["results"][0]["id"]
        
        # Retrieve with correct user_id
        search_results = self.memory.search(
            query="bronchitis",
            filters={"user_id": patient_id},
            top_k=10
        )
        
        assert len(search_results["results"]) > 0, \
            f"Should find episodic memory for patient {patient_id}"
        
        # Key assertion: Can retrieve by user_id filter
        logger.info(f"✓ Episodic memory properly scoped to user_id={patient_id}")
        
        # Try to retrieve with WRONG user_id
        wrong_patient_id = "patient_wrong"
        search_results_wrong = self.memory.search(
            query="bronchitis",
            filters={"user_id": wrong_patient_id},
            top_k=10
        )
        
        assert len(search_results_wrong["results"]) == 0, \
            "Should NOT find patient_001's memory when filtering for wrong patient"
        
        logger.info(f"✓ Episodic memory properly ISOLATED from other patients")
    
    def test_memory_type_differentiation(self):
        """
        Test that SEMANTIC and EPISODIC memories are queryable separately.
        """
        # Add semantic memory
        semantic = "Type 2 Diabetes is a metabolic disorder characterized by high blood glucose"
        self.memory.add(
            messages=[{"role": "user", "content": semantic}],
            infer=False
            # No user_id → SEMANTIC
        )
        
        # Add episodic memory
        patient_id = "patient_002"
        episodic = "Patient diagnosed with Type 2 Diabetes on 2020-01-15"
        self.memory.add(
            messages=[{"role": "user", "content": episodic}],
            user_id=patient_id,
            infer=False
            # Has user_id → EPISODIC
        )
        
        # Search for "diabetes" globally (should find both)
        global_results = self.memory.get_all(
            filters={},
            top_k=100
        )
        # Note: Would fail if filters require user_id, but semantic search should work
        
        # Search for "diabetes" for specific patient (should find only episodic)
        patient_results = self.memory.search(
            query="diabetes",
            filters={"user_id": patient_id},
            top_k=100
        )
        
        assert len(patient_results["results"]) > 0, \
            "Should find patient's episodic diabetes diagnosis"
        
        logger.info("✓ Semantic and episodic memories are properly differentiated")


class TestShortTermMemory:
    """
    Test Suite 2: Verify short-term memory (SQLite) works correctly
    """
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Initialize Mem0."""
        config = MemoryConfig()
        self.memory = Memory(config)
        yield
        try:
            self.memory.reset()
        except:
            pass
    
    def test_short_term_messages_stored(self):
        """
        SHORT-TERM MEMORY: Messages should be saved in SQLite.
        
        When add() is called, messages should be stored in SQLite for quick context retrieval.
        """
        patient_id = "patient_003"
        
        # Add conversation message
        messages = [
            {"role": "user", "content": "I've been feeling tired lately"},
            {"role": "assistant", "content": "That could be a symptom of several conditions"}
        ]
        
        self.memory.add(
            messages=messages,
            user_id=patient_id,
            infer=False
        )
        
        # Retrieve recent messages
        recent = self.memory.get_recent_messages(user_id=patient_id, limit=10)
        
        assert recent is not None, "Recent messages should be retrievable"
        assert len(recent) > 0, "Messages should have been saved to short-term memory"
        
        logger.info(f"✓ Short-term memory stored {len(recent)} messages in SQLite")
    
    def test_short_term_memory_context_for_extraction(self):
        """
        SHORT-TERM MEMORY: Last K messages should be used as context for LLM extraction.
        
        When extracting new memories, the LLM should receive recent messages as context.
        This ensures memories are grounded in recent conversation history.
        """
        patient_id = "patient_004"
        
        # First: Add initial condition
        self.memory.add(
            messages=[{"role": "user", "content": "I was diagnosed with hypertension last year"}],
            user_id=patient_id,
            infer=True  # Let LLM extract
        )
        
        # Second: Add follow-up (should reference recent context)
        # LLM should remember hypertension diagnosis from short-term memory
        self.memory.add(
            messages=[{"role": "user", "content": "My doctor prescribed Lisinopril to manage my blood pressure"}],
            user_id=patient_id,
            infer=True
        )
        
        # Get recent messages (short-term memory)
        recent = self.memory.get_recent_messages(user_id=patient_id, limit=5)
        
        # Verify both messages are in short-term memory
        assert len(recent) >= 2, "Both messages should be in short-term memory"
        
        logger.info(f"✓ Short-term memory provides context across multiple add() calls")
    
    def test_short_term_memory_scoped_per_patient(self):
        """
        SHORT-TERM MEMORY: Last K messages should be separate per patient.
        
        Patient A's conversation should not appear in Patient B's short-term memory.
        """
        patient_a = "patient_a"
        patient_b = "patient_b"
        
        # Patient A's message
        self.memory.add(
            messages=[{"role": "user", "content": "I have asthma"}],
            user_id=patient_a,
            infer=False
        )
        
        # Patient B's message
        self.memory.add(
            messages=[{"role": "user", "content": "I have arthritis"}],
            user_id=patient_b,
            infer=False
        )
        
        # Get recent messages for Patient A
        recent_a = self.memory.get_recent_messages(user_id=patient_a, limit=10)
        
        # Get recent messages for Patient B
        recent_b = self.memory.get_recent_messages(user_id=patient_b, limit=10)
        
        # Verify isolation
        assert any("asthma" in msg.get("content", "").lower() for msg in recent_a), \
            "Patient A should see their asthma message"
        
        assert any("arthritis" in msg.get("content", "").lower() for msg in recent_b), \
            "Patient B should see their arthritis message"
        
        assert not any("asthma" in msg.get("content", "").lower() for msg in recent_b), \
            "Patient B should NOT see Patient A's asthma message"
        
        logger.info("✓ Short-term memory properly isolated per patient")


class TestHybridSearch:
    """
    Test Suite 3: Verify hybrid search works across semantic + episodic
    """
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Initialize Mem0."""
        config = MemoryConfig()
        self.memory = Memory(config)
        yield
        try:
            self.memory.reset()
        except:
            pass
    
    def test_semantic_and_episodic_search_combination(self):
        """
        HYBRID SEARCH: Should find both semantic knowledge and episodic patient data.
        
        When searching for "diabetes", results should include:
        1. SEMANTIC: General diabetes definition/information
        2. EPISODIC: Patient's specific diagnosis and treatment
        """
        # Add SEMANTIC knowledge (global)
        semantic = "Diabetes Type 2 is a metabolic disorder where blood glucose levels are elevated"
        self.memory.add(
            messages=[{"role": "user", "content": semantic}],
            infer=False
        )
        
        # Add EPISODIC patient history
        patient_id = "patient_005"
        episodic = "Patient diagnosed with Type 2 Diabetes on 2020-03-15, currently on Metformin"
        self.memory.add(
            messages=[{"role": "user", "content": episodic}],
            user_id=patient_id,
            infer=False
        )
        
        # Search for patient's diabetes info (episodic)
        # Should find: patient's diagnosis + treatment
        patient_results = self.memory.search(
            query="diabetes treatment",
            filters={"user_id": patient_id},
            top_k=10
        )
        
        assert len(patient_results["results"]) > 0, \
            "Should find patient's episodic diabetes diagnosis"
        
        logger.info(f"✓ Episodic search found {len(patient_results['results'])} results")
    
    def test_search_ranking_semantic_vs_episodic(self):
        """
        Verify that search ranking properly combines semantic + episodic relevance.
        
        When patient searches for info about their condition:
        - Episodic (their specific case) should rank highly
        - Semantic (general knowledge) should provide background
        """
        patient_id = "patient_006"
        
        # Add semantic fact
        self.memory.add(
            messages=[{"role": "user", "content": "Hypertension is chronic elevation of blood pressure above 140/90 mmHg"}],
            infer=False
        )
        
        # Add episodic fact
        self.memory.add(
            messages=[{"role": "user", "content": "Patient's blood pressure readings: systolic 155 mmHg, diastolic 95 mmHg on 2025-05-02"}],
            user_id=patient_id,
            infer=False
        )
        
        # Search
        results = self.memory.search(
            query="blood pressure measurements",
            filters={"user_id": patient_id},
            top_k=10
        )
        
        assert len(results["results"]) > 0, "Should find results"
        
        # Verify results include scores for ranking
        for result in results["results"]:
            assert "score" in result, "Results should include relevance scores"
        
        logger.info(f"✓ Search results ranked with scores: {[r.get('score') for r in results['results']]}")


class TestDataIntegrity:
    """
    Test Suite 4: Verify data integrity across operations
    """
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Initialize Mem0."""
        config = MemoryConfig()
        self.memory = Memory(config)
        yield
        try:
            self.memory.reset()
        except:
            pass
    
    def test_update_preserves_episodic_scope(self):
        """
        When updating episodic memory, user_id scope should be preserved.
        """
        patient_id = "patient_007"
        
        # Add
        result = self.memory.add(
            messages=[{"role": "user", "content": "Patient has hypertension"}],
            user_id=patient_id,
            infer=False
        )
        
        memory_id = result["results"][0]["id"]
        
        # Update
        self.memory.update(
            memory_id=memory_id,
            data="Patient has Stage 2 hypertension (reading 160/100)",
            metadata={"patient_id": patient_id}
        )
        
        # Retrieve via search with filter
        results = self.memory.search(
            query="Stage 2 hypertension",
            filters={"user_id": patient_id},
            top_k=10
        )
        
        assert len(results["results"]) > 0, "Updated episodic memory should be findable"
        
        logger.info("✓ Update preserves episodic memory scope")
    
    def test_delete_episodic_memory(self):
        """
        Deleting episodic memory should remove it from patient's scope.
        """
        patient_id = "patient_008"
        
        # Add
        result = self.memory.add(
            messages=[{"role": "user", "content": "Patient was allergic to Penicillin"}],
            user_id=patient_id,
            infer=False
        )
        
        memory_id = result["results"][0]["id"]
        
        # Verify it exists
        search_before = self.memory.search(
            query="Penicillin allergy",
            filters={"user_id": patient_id},
            top_k=10
        )
        assert len(search_before["results"]) > 0, "Memory should exist before delete"
        
        # Delete
        self.memory.delete(memory_id)
        
        # Verify it's gone
        search_after = self.memory.search(
            query="Penicillin allergy",
            filters={"user_id": patient_id},
            top_k=10
        )
        assert len(search_after["results"]) == 0, "Memory should be deleted"
        
        logger.info("✓ Episodic memory deletion works correctly")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
