# Healthcare Memory System - API Key Setup Guide

## Quick Setup (2 minutes)

The healthcare memory system requires an **OpenAI API key** for embeddings.

### Step 1: Get Your API Key

1. Go to: **https://platform.openai.com/api-keys**
2. Sign in with your OpenAI account (create one if needed)
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-`)
5. Save it somewhere safe (you won't see it again!)

### Step 2: Set Environment Variable

Choose your operating system:

#### **Windows PowerShell** (Recommended)
```powershell
# Temporary (current session only)
$env:OPENAI_API_KEY = "sk-your-key-here"

# Verify it's set
$env:OPENAI_API_KEY
# Output: sk-your-key-here

# Then run quickstart
.\venv\Scripts\python.exe quickstart.py
```

#### **Windows Command Prompt**
```cmd
set OPENAI_API_KEY=sk-your-key-here
python quickstart.py
```

#### **Linux/Mac Terminal**
```bash
export OPENAI_API_KEY="sk-your-key-here"
python quickstart.py
```

### Step 3: Permanent Setup (Optional)

To make the key permanent so you don't have to set it every time:

#### **Windows PowerShell (Permanent)**
```powershell
# Set permanently for current user
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY","sk-your-key-here","User")

# Restart your terminal to apply
# Then verify:
$env:OPENAI_API_KEY
```

#### **Windows Edit Environment Variables (GUI)**
1. Press `Win + X`, select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `OPENAI_API_KEY`
6. Variable value: `sk-your-key-here`
7. Click OK, restart terminal

#### **Linux/Mac (~/.bashrc or ~/.zshrc)**
```bash
# Add to end of ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-your-key-here"

# Then reload:
source ~/.bashrc
# or
source ~/.zshrc
```

### Step 4: Verify Setup

```bash
# Check if key is set
echo $env:OPENAI_API_KEY  # PowerShell
echo $OPENAI_API_KEY      # Linux/Mac

# Should output: sk-...
```

### Step 5: Run Demo

```bash
python quickstart.py
```

Expected output:
```
================================================================================
HEALTHCARE MEMORY SYSTEM - QUICK START
================================================================================

[Step 1] Importing modules...
✓ Imports successful

[Step 2] Initializing healthcare memory system...
✓ System initialized

[Step 3] Loading SEMANTIC memory (MedQuAD medical knowledge)...
...
```

---

## Troubleshooting

### Error: "OPENAI_API_KEY environment variable"

**Problem**: The environment variable is not set.

**Solution**: 
1. Verify you've set it: `echo $env:OPENAI_API_KEY`
2. If empty, set it again (follow Step 2 above)
3. Make sure you're using the right command for your shell

### Error: "Invalid API key"

**Problem**: The key is incorrect or expired.

**Solution**:
1. Get a fresh key from https://platform.openai.com/api-keys
2. Verify the key starts with `sk-`
3. Make sure you copied it completely (no spaces)
4. Set it again and retry

### Error: "Quota exceeded"

**Problem**: You've used all your API credits.

**Solution**:
1. Check your usage at https://platform.openai.com/account/usage
2. Set up billing or request a usage limit increase
3. Consider using a local embedding model instead (see Alternative below)

---

## Alternative: Use Local Embedding Model (Free)

If you don't have an OpenAI API key or don't want to use it, you can use a local embedding model:

### Option 1: FastEmbed (Recommended, Free)

Modify `healthcare_memory_system.py`:

```python
from mem0.configs.base import MemoryConfig
from mem0.embeddings.fastembedding import FastEmbeddingConfig

config = MemoryConfig(
    embedder=FastEmbeddingConfig(
        provider="fastembed",
        config={"model": "BAAI/bge-small-en-v1.5"}
    ),
    ...
)
```

Then run without API key:
```bash
python quickstart.py  # No OPENAI_API_KEY needed!
```

### Option 2: Ollama (Free, Requires Docker)

Setup Ollama:
```bash
# Download from https://ollama.ai
ollama pull mistral  # or another model
ollama serve
```

Then modify `healthcare_memory_system.py` to use Ollama.

---

## Cost Considerations

### OpenAI Pricing
- **Embeddings (text-embedding-3-small)**: $0.02 per 1M tokens
- **Average document**: ~100-300 tokens
- **Healthcare system**: 
  - Load 1000 MedQuAD: ~$0.02-0.05
  - Load 1 patient (50 records): ~$0.001-0.002
  - Search: ~$0.0001 per query

**Estimated cost**: $1-5/month for typical usage

### Free Alternatives
- **FastEmbed**: Free, runs locally (recommended for development)
- **HuggingFace**: Free embeddings model
- **Ollama**: Free, runs locally

---

## Quick Reference

```bash
# Set key (temporary)
$env:OPENAI_API_KEY = "sk-your-key"

# Verify key is set
$env:OPENAI_API_KEY

# Run demo
python quickstart.py

# Run tests
pytest test_healthcare_integration.py -v

# Run validation
python validate_architecture.py
```

---

## Questions?

1. **How do I get a free OpenAI key?**
   - OpenAI offers $5 free credits for new accounts (valid for 3 months)
   - Visit: https://platform.openai.com/account/billing/overview

2. **Is my API key secure?**
   - Keep it private! Never commit it to git
   - Use environment variables (not hardcoded in code)
   - Regenerate if you think it's compromised

3. **Can I use a different LLM provider?**
   - Yes! Mem0 supports: Azure OpenAI, Anthropic, Gemini, Ollama, and more
   - See: https://docs.mem0.ai

4. **What if I don't want to use OpenAI?**
   - Use FastEmbed (local, free, good quality)
   - Use Ollama (free, self-hosted)
   - See Alternative section above

---

## Next Steps

Once API key is set:

1. Run demo: `python quickstart.py`
2. Run tests: `pytest test_healthcare_integration.py -v`
3. Read guide: `HEALTHCARE_INTEGRATION_README.md`
4. Build your app!

Good luck! 🚀
