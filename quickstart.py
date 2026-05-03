#!/usr/bin/env python3
"""
Healthcare Memory System - Quick Start Guide
============================================

Run this script to see the complete integration in action.

SETUP REQUIRED:
    Set OPENAI_API_KEY environment variable before running:
    
    Windows PowerShell:
        $env:OPENAI_API_KEY = "sk-your-api-key-here"
    
    Windows Command Prompt:
        set OPENAI_API_KEY=sk-your-api-key-here
    
    Linux/Mac:
        export OPENAI_API_KEY="sk-your-api-key-here"
    
    Get your API key from: https://platform.openai.com/api-keys
"""

import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("\n" + "="*80)
        logger.error("OPENAI_API_KEY NOT SET")
        logger.error("="*80)
        logger.error("\nYou need to set the OPENAI_API_KEY environment variable.")
        logger.error("\nTo get your API key:")
        logger.error("  1. Go to: https://platform.openai.com/api-keys")
        logger.error("  2. Click 'Create new secret key'")
        logger.error("  3. Copy the key")
        logger.error("\nTo set it in PowerShell:")
        logger.error('  $env:OPENAI_API_KEY = "sk-..."')
        logger.error("\nTo set it permanently (PowerShell):")
        logger.error('  [Environment]::SetEnvironmentVariable("OPENAI_API_KEY","sk-...", "User")')
        logger.error("  (Then restart your terminal)")
        logger.error("\n" + "="*80 + "\n")
        return False
    return True


def main():
    """Execute quick start demo."""
    
    logger.info("="*80)
    logger.info("HEALTHCARE MEMORY SYSTEM - QUICK START")
    logger.info("="*80)
    
    # Check for API key first
    if not check_api_key():
        return False
    
    try:
        # Step 1: Import
        logger.info("\n[Step 1] Importing modules...")
        from healthcare_memory_system import HealthcareMemorySystem
        from healthcare_integration import HealthcareDatasetLoader
        logger.info("✓ Imports successful")
        
        # Step 2: Initialize
        logger.info("\n[Step 2] Initializing healthcare memory system...")
        logger.info("  (This will initialize Mem0 with OpenAI embeddings)")
        system = HealthcareMemorySystem()
        logger.info("✓ System initialized")
        
        # Step 3: Load semantic memory (medical knowledge)
        logger.info("\n[Step 3] Loading SEMANTIC memory (MedQuAD medical knowledge)...")
        logger.info("  This adds global medical facts (not patient-specific)")
        semantic_result = system.add_semantic_medical_knowledge(limit=5)
        logger.info(f"✓ Added {semantic_result['added']} semantic medical memories")
        
        # Step 4: Get a sample patient
        logger.info("\n[Step 4] Loading sample patient from Synthea...")
        loader = system.loader
        patients = loader.load_synthea_patients(limit=1)
        
        if not patients:
            logger.error("✗ No patients found in Synthea dataset")
            return False
        
        patient = patients[0]
        patient_id = patient["PATIENT_ID"]
        logger.info(f"✓ Patient: {patient['FIRST']} {patient['LAST']}")
        logger.info(f"  ID: {patient_id}")
        logger.info(f"  DOB: {patient['BIRTHDATE']}")
        
        # Step 5: Load episodic memory (patient history)
        logger.info("\n[Step 5] Loading EPISODIC memory (Synthea patient history)...")
        logger.info("  This adds patient-specific diagnoses, medications, encounters")
        episodic_result = system.add_patient_history(patient_id)
        logger.info(f"✓ Added episodic memories for patient:")
        logger.info(f"    - Conditions: {episodic_result['conditions']}")
        logger.info(f"    - Medications: {episodic_result['medications']}")
        logger.info(f"    - Encounters: {episodic_result['encounters']}")
        logger.info(f"    - Total: {episodic_result['total_added']}")
        
        # Step 6: Add short-term context
        logger.info("\n[Step 6] Adding SHORT-TERM conversation context (SQLite)...")
        conversation = [
            {
                "role": "user",
                "content": "I've been experiencing chest discomfort lately"
            },
            {
                "role": "assistant",
                "content": "When did this discomfort start, and how severe is it on a scale of 1-10?"
            },
            {
                "role": "user",
                "content": "It started about a week ago, and it's about a 5 or 6"
            }
        ]
        
        short_term_result = system.add_patient_short_term_context(patient_id, conversation)
        logger.info(f"✓ Short-term memory context added:")
        logger.info(f"    - Messages stored: {short_term_result['short_term_count']}")
        logger.info(f"    - Extracted memories: {short_term_result['extracted_memories']}")
        
        # Step 7: Search (combines all memory types)
        logger.info("\n[Step 7] Searching patient knowledge (HYBRID search)...")
        logger.info("  This combines SEMANTIC + EPISODIC + SHORT-TERM in ranking")
        
        search_result = system.search_patient_knowledge(
            patient_id=patient_id,
            query="chest pain cardiac medication",
            top_k=5,
            include_semantic=True
        )
        
        logger.info(f"✓ Search results:")
        logger.info(f"    - Episodic (patient-specific): {len(search_result['episodic'])}")
        logger.info(f"    - Semantic (general knowledge): {len(search_result['semantic'])}")
        logger.info(f"    - Total: {search_result['combined_count']}")
        
        if search_result['episodic']:
            logger.info("\n  Top Episodic Results:")
            for i, mem in enumerate(search_result['episodic'][:3], 1):
                score = mem.get('score', 0)
                text = mem.get('memory', '')[:60]
                logger.info(f"    [{i}] (Score: {score:.2f}) {text}...")
        
        # Step 8: Get patient full history
        logger.info("\n[Step 8] Retrieving patient's full memory history...")
        history = system.get_patient_full_history(patient_id, top_k=10)
        logger.info(f"✓ Patient has {history['memory_count']} memories on record")
        
        if history['memories']:
            logger.info("\n  Sample Memories:")
            for mem in history['memories'][:3]:
                text = mem.get('memory', '')[:60]
                logger.info(f"    - {text}...")
        
        # Step 9: Get recent messages
        logger.info("\n[Step 9] Retrieving recent conversation context...")
        recent = system.get_patient_recent_context(patient_id, limit=5)
        logger.info(f"✓ Retrieved {len(recent)} recent messages")
        
        for msg in recent[:2]:
            role = msg.get('role', '').upper()
            content = msg.get('content', '')[:50]
            logger.info(f"    {role}: {content}...")
        
        # Success
        logger.info("\n" + "="*80)
        logger.info("✓ QUICK START DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\nNext Steps:")
        logger.info("  1. Run: python validate_architecture.py")
        logger.info("     (Validates semantic/episodic separation)")
        logger.info("\n  2. Run: pytest test_healthcare_integration.py -v")
        logger.info("     (Full test suite with 15+ tests)")
        logger.info("\n  3. See: HEALTHCARE_INTEGRATION_README.md")
        logger.info("     (Complete documentation)")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
