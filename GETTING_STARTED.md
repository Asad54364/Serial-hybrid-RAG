# Quick Setup Instructions

## You're Missing the OpenAI API Key!

The quickstart.py demo failed because it needs an OpenAI API key. Here's what to do:

### Option 1: Use the NO-API-KEY Demo (Recommended First)

This shows the architecture without needing any API keys:

```bash
python demo.py
```

This will show you:
- How MedQuAD loads as semantic memory
- How Synthea loads as episodic memory  
- Data formatting and structure
- Architecture overview

### Option 2: Get an API Key and Run Full Demo

Follow these steps:

1. **Get your OpenAI API key:**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (starts with `sk-`)

2. **Set the environment variable in PowerShell:**
   ```powershell
   $env:OPENAI_API_KEY = "sk-your-key-here"
   ```

3. **Verify it's set:**
   ```powershell
   $env:OPENAI_API_KEY
   # Should output: sk-...
   ```

4. **Run the full demo:**
   ```bash
   python quickstart.py
   ```

### Option 3: Make the API Key Permanent

So you don't have to set it every time:

```powershell
# Set permanently for current user
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY","sk-your-key-here","User")

# Restart your terminal
# Then the key will be automatically available
```

### Option 4: Use a Free Alternative (FastEmbed)

No API key needed! Local embeddings model:

See SETUP_API_KEY.md "Alternative: Use Local Embedding Model" section.

---

## Files Available

- **demo.py** - Run this first (no API key needed!)
- **quickstart.py** - Full demo (needs API key)
- **SETUP_API_KEY.md** - Complete setup guide
- **HEALTHCARE_INTEGRATION_README.md** - Usage documentation
- **ARCHITECTURE_DIAGRAMS.md** - Visual reference

---

## Quick Commands

```bash
# Try the no-key demo first
python demo.py

# After setting OPENAI_API_KEY:
python quickstart.py

# Run all tests (after API key setup)
pytest test_healthcare_integration.py -v

# Validate architecture (after API key setup)
python validate_architecture.py

# Read the documentation
HEALTHCARE_INTEGRATION_README.md
```

---

Need help? See SETUP_API_KEY.md for detailed instructions.
