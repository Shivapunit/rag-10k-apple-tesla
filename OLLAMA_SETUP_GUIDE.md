📚 OLLAMA SETUP & TROUBLESHOOTING GUIDE

═══════════════════════════════════════════════════════════════════════════════

🚀 QUICK SETUP (5 MINUTES)

1. DOWNLOAD OLLAMA
   ────────────────
   Visit: https://ollama.ai

   Select your OS:
   • Mac (Intel/Apple Silicon)
   • Linux
   • Windows

2. INSTALL
   ────────

   Mac:
     • Open downloaded .dmg
     • Drag Ollama to Applications
     • Run from Applications folder
     • It auto-starts in background

   Linux:
     curl https://ollama.ai/install.sh | sh
     systemctl enable ollama

   Windows:
     • Run .exe installer
     • Follow prompts
     • Restart your terminal

3. START MISTRAL MODEL
   ───────────────────
   Open terminal and run:

   ```bash
   ollama run mistral
   ```

   This will:
     • Download Mistral-7B (~4GB) - first time only
     • Start Ollama server
     • Keep running in background
     • Listen on http://localhost:11434

4. VERIFY RUNNING
   ───────────────
   In another terminal:

   ```bash
   curl http://localhost:11434/api/tags
   ```

   Should return JSON with models list.

5. RUN STREAMLIT APP
   ──────────────────
   ```bash
   streamlit run app.py
   ```

   Opens at: http://localhost:8501

═══════════════════════════════════════════════════════════════════════════════

🐛 TROUBLESHOOTING

PROBLEM: "Connection refused" or "Cannot connect to localhost:11434"
──────────────────────────────────────────────────────────────────
Solution:
  1. Check if Ollama is running
  2. Terminal: ollama run mistral
  3. Wait 5-10 seconds for server to start
  4. Refresh Streamlit app (F5)

Check:
  curl http://localhost:11434/api/tags

Should return HTTP 200 with model data.


PROBLEM: "Timeout" or "Connection slow"
───────────────────────────────────────
Solution:
  1. Ollama might be downloading model
  2. First run takes 5-10 minutes (4GB download)
  3. Wait and check progress in terminal
  4. Don't close the ollama terminal

Or:
  1. System might be low on resources
  2. Close other applications
  3. Make sure 4GB RAM available
  4. Try again


PROBLEM: "Model not found" or "No models loaded"
────────────────────────────────────────────────
Solution:
  1. Ollama hasn't downloaded model yet
  2. Run: ollama run mistral
  3. Wait for download to complete
  4. You'll see progress: "100%"
  5. Then try Streamlit app again


PROBLEM: "Port 11434 already in use"
────────────────────────────────────
Solution:

  Mac/Linux:
    lsof -i :11434
    kill -9 <PID>

  Windows:
    netstat -ano | findstr :11434
    taskkill /PID <PID> /F

  Then restart Ollama.


PROBLEM: App shows "Ollama LLM Server Not Found"
────────────────────────────────────────────────
What this means:
  • Ollama server is NOT running
  • OR not accessible at localhost:11434
  • OR blocking firewall

Fix:
  1. Ensure terminal with "ollama run mistral" is open
  2. Check terminal shows "listening on"
  3. Try: curl http://localhost:11434/api/tags
  4. If curl fails, Ollama not running
  5. Restart: ollama run mistral
  6. Wait 10 seconds
  7. Refresh Streamlit (F5)


PROBLEM: Streamlit app crashes or freezes
──────────────────────────────────────────
Solution:
  1. Check Ollama terminal for errors
  2. Restart Ollama: Ctrl+C then ollama run mistral
  3. Restart Streamlit: Ctrl+C then streamlit run app.py
  4. If still slow: Ollama might be generating response
  5. Wait 30 seconds before closing


PROBLEM: "Too many open requests" or "Connection pool exhausted"
────────────────────────────────────────────────────────────────
Solution:
  1. Close other Ollama connections
  2. Restart both Ollama and Streamlit
  3. Only one Streamlit tab open
  4. Avoid rapid clicking


═══════════════════════════════════════════════════════════════════════════════

✅ VERIFICATION CHECKLIST

After installing:
  ☐ Ollama installed and in PATH
  ☐ Terminal: ollama --version (shows version)
  ☐ Terminal: ollama run mistral (downloads and starts)
  ☐ Server shows: "listening on 127.0.0.1:11434"
  ☐ curl returns: {"models": [{"name": "mistral:latest", ...}]}
  ☐ Streamlit app shows: "✅ Ollama is running"
  ☐ Can ask questions and get answers
  ☐ Responses include sources [Document, Item, Page]


═══════════════════════════════════════════════════════════════════════════════

🌐 CLOUD ALTERNATIVES (No Ollama Needed)

If you can't install Ollama:

1. Google Colab (Recommended)
   ├─ Link: notebooks/rag_colab.ipynb
   ├─ Click: "Open in Colab"
   ├─ Run: All cells
   └─ Result: Full system works in browser

2. Command Line
   ├─ Command: python test_runner.py
   ├─ Output: test_results.json
   └─ Works: Anywhere (no Ollama needed)

3. Cloud VM with Ollama
   ├─ AWS EC2, Google Cloud, Azure VM
   ├─ Install Ollama on VM
   ├─ Run Streamlit
   └─ Access via cloud URL


═══════════════════════════════════════════════════════════════════════════════

📋 PERFORMANCE EXPECTATIONS

First run (downloading model):
  • Mistral download: 3-10 minutes
  • Depends on internet speed
  • One-time only
  • Model size: ~4GB

Normal operation (after model downloaded):
  • Query processing: 3-15 seconds
  • Retrieval: <100ms
  • LLM generation: 2-10 seconds
  • Total: ~5-15 seconds per question

System requirements:
  • RAM: 8GB+ (16GB recommended)
  • Disk: 5GB+ free space
  • CPU: Any modern CPU (GPU optional)
  • Internet: For initial model download only


═══════════════════════════════════════════════════════════════════════════════

🔗 HELPFUL LINKS

Ollama Documentation:
  https://github.com/ollama/ollama

Model Library:
  https://ollama.ai/library

Mistral Model:
  https://ollama.ai/library/mistral

Forum/Issues:
  https://github.com/ollama/ollama/issues


═══════════════════════════════════════════════════════════════════════════════

QUICK START COMMAND (Copy & Paste)

Mac/Linux:
  # Terminal 1
  ollama run mistral

  # Terminal 2
  pip install -r requirements.txt
  streamlit run app.py

Windows:
  # Terminal 1
  ollama run mistral

  # Terminal 2
  pip install -r requirements.txt
  streamlit run app.py

Both: Open browser to http://localhost:8501

═══════════════════════════════════════════════════════════════════════════════

