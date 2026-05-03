#!/usr/bin/env python3
"""
Healthcare Memory System - Architecture Demo (No API Key Required!)
===================================================================

This demo shows the healthcare memory system architecture without needing
an OpenAI API key. It demonstrates data loading and formatting.

To run the FULL demo WITH memory operations, see quickstart.py and
follow the SETUP_API_KEY.md guide.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate healthcare memory architecture."""
    
    logger.info("="*80)
    logger.info("HEALTHCARE MEMORY SYSTEM - ARCHITECTURE DEMO")
    logger.info("(No API Key Required!)")
    logger.info("="*80)
    
    try:
        # Step 1: Import dataset loader
        logger.info("\n[Step 1] Importing dataset loader...")
        from healthcare_integration import HealthcareDatasetLoader
        logger.info("✓ Dataset loader imported")
        
        # Step 2: Initialize loader
        logger.info("\n[Step 2] Initializing dataset loader...")
        loader = HealthcareDatasetLoader()
        logger.info("✓ Loader initialized")
        
        # Step 3: Load semantic data (MedQuAD)
        logger.info("\n[Step 3] Loading SEMANTIC memory (MedQuAD)...")
        logger.info("  Loading 5 medical QA pairs as examples...")
        
        medquad_data = loader.load_medquad(limit=5)
        logger.info(f"✓ Loaded {len(medquad_data)} MedQuAD entries")
        
        if medquad_data:
            logger.info("\n  Example Semantic Memories:")
            for i, qa in enumerate(medquad_data[:3], 1):
                formatted = loader.format_semantic_memory(qa)
                logger.info(f"\n  [{i}] {qa.get('focus_area', 'Medical Knowledge')}")
                logger.info(f"      Q: {qa.get('question', '')[:60]}...")
                logger.info(f"      → Stored as: {formatted[:70]}...")
        
        # Step 4: Load episodic data (Synthea)
        logger.info("\n[Step 4] Loading EPISODIC memory (Synthea patient data)...")
        logger.info("  Loading sample patients...")
        
        patients = loader.load_synthea_patients(limit=3)
        logger.info(f"✓ Loaded {len(patients)} patient records")
        
        if patients:
            logger.info("\n  Example Episodic Memories (Patient Demographics):")
            for i, patient in enumerate(patients[:2], 1):
                logger.info(f"\n  [{i}] Patient: {patient.get('FIRST', '')} {patient.get('LAST', '')}")
                logger.info(f"      ID: {patient.get('PATIENT_ID', '')}")
                logger.info(f"      DOB: {patient.get('BIRTHDATE', '')}")
                logger.info(f"      Status: {patient.get('MARITAL', '')}")
        
        # Step 5: Load patient conditions
        if patients:
            logger.info("\n[Step 5] Loading patient conditions...")
            patient_id = patients[0]["PATIENT_ID"]
            logger.info(f"  Loading conditions for patient: {patient_id[:8]}...")
            
            conditions = loader.load_synthea_conditions(patient_id, limit=5)
            logger.info(f"✓ Loaded {len(conditions)} condition records")
            
            if conditions:
                logger.info("\n  Example Conditions (Episodic Memories):")
                for i, cond in enumerate(conditions[:3], 1):
                    formatted = loader.format_episodic_condition(cond)
                    logger.info(f"\n  [{i}] {cond.get('DESCRIPTION', 'Condition')}")
                    logger.info(f"      Date: {cond.get('START', '')}")
                    logger.info(f"      → Stored as: {formatted[:70]}...")
        
        # Step 6: Load medications
        if patients:
            logger.info("\n[Step 6] Loading patient medications...")
            patient_id = patients[0]["PATIENT_ID"]
            logger.info(f"  Loading medications for patient: {patient_id[:8]}...")
            
            medications = loader.load_synthea_medications(patient_id, limit=5)
            logger.info(f"✓ Loaded {len(medications)} medication records")
            
            if medications:
                logger.info("\n  Example Medications (Episodic Memories):")
                for i, med in enumerate(medications[:3], 1):
                    formatted = loader.format_episodic_medication(med)
                    logger.info(f"\n  [{i}] {med.get('DESCRIPTION', 'Medication')}")
                    logger.info(f"      Date: {med.get('START', '')}")
                    logger.info(f"      → Stored as: {formatted[:70]}...")
        
        # Step 7: Architecture summary
        logger.info("\n[Step 7] Architecture Summary...")
        logger.info("\n  THREE MEMORY LAYERS:")
        logger.info("  ├─ SEMANTIC (Global Medical Knowledge)")
        logger.info("  │  ├─ Source: MedQuAD")
        logger.info("  │  ├─ Scope: No user_id (global)")
        logger.info("  │  └─ Storage: Vector embeddings")
        logger.info("  │")
        logger.info("  ├─ EPISODIC (Patient-Specific History)")
        logger.info("  │  ├─ Source: Synthea")
        logger.info("  │  ├─ Scope: user_id = patient_id (isolated)")
        logger.info("  │  └─ Storage: Vector embeddings")
        logger.info("  │")
        logger.info("  └─ SHORT-TERM (Recent Conversation)")
        logger.info("     ├─ Source: Conversation messages")
        logger.info("     ├─ Scope: per-patient session")
        logger.info("     └─ Storage: SQLite")
        
        # Step 8: Memory isolation guarantee
        logger.info("\n[Step 8] Patient Isolation Guarantee...")
        logger.info("\n  How it works:")
        logger.info("  • Semantic memories have NO user_id → accessible to all patients")
        logger.info("  • Episodic memories HAVE user_id → isolated per patient")
        logger.info("  • Queries use filters to enforce:")
        logger.info("    - search(query, filters={}) → semantic only")
        logger.info("    - search(query, filters={'user_id': 'A'}) → episodic + semantic for A")
        logger.info("  • Result: Patient A CANNOT see Patient B's data")
        
        # Step 9: Next steps
        logger.info("\n" + "="*80)
        logger.info("✓ ARCHITECTURE DEMO COMPLETE!")
        logger.info("="*80)
        
        logger.info("\n📋 What You Learned:")
        logger.info("  1. MedQuAD loads as SEMANTIC (global medical knowledge)")
        logger.info("  2. Synthea loads as EPISODIC (patient-specific histories)")
        logger.info("  3. Data formatted as memory statements for LLM extraction")
        logger.info("  4. Patient isolation enforced via user_id filtering")
        logger.info("  5. Three memory layers store different types of data")
        
        logger.info("\n🚀 Next Steps:")
        logger.info("  1. To run FULL demo with Mem0 memory operations:")
        logger.info("     → Follow: SETUP_API_KEY.md")
        logger.info("     → Then run: python quickstart.py")
        logger.info("\n  2. To run tests:")
        logger.info("     → pytest test_healthcare_integration.py -v")
        logger.info("\n  3. To validate architecture:")
        logger.info("     → python validate_architecture.py")
        logger.info("\n  4. To understand the system:")
        logger.info("     → Read: HEALTHCARE_INTEGRATION_README.md")
        logger.info("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
