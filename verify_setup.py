"""
Verify Installation Script
Check if all dependencies are installed correctly
"""

import sys
import subprocess
from pathlib import Path

print("\n" + "=" * 80)
print("RAG SYSTEM - DEPENDENCY VERIFICATION")
print("=" * 80 + "\n")

# Check Python version
print("[1/5] Checking Python version...")
py_version = sys.version_info
print(f"  Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version >= (3, 10):
    print("  ✅ OK (3.10+)\n")
else:
    print("  ❌ FAIL (need 3.10+)\n")
    sys.exit(1)

# Check key packages
print("[2/5] Checking Python packages...")
packages = {
    'langchain': 'LLM orchestration',
    'langchain_community': 'Community integrations',
    'pypdf': 'PDF parsing',
    'faiss': 'Vector database',
    'sentence_transformers': 'Embeddings',
    'streamlit': 'Web UI',
}

failed = []
for package, description in packages.items():
    try:
        __import__(package)
        print(f"  ✅ {package:30} {description}")
    except ImportError:
        print(f"  ❌ {package:30} {description}")
        failed.append(package)

if failed:
    print(f"\n  Missing packages: {', '.join(failed)}")
    print(f"\n  Install with: pip install -r requirements.txt\n")
    sys.exit(1)
else:
    print("  ✅ All packages installed\n")

# Check file structure
print("[3/5] Checking project structure...")
required_files = [
    'rag_pipeline.py',
    'ingest.py',
    'app.py',
    'test_runner.py',
    'requirements.txt',
    'README.md',
    'design_report.md',
    'GETTING_STARTED.md',
]

missing_files = []
for file in required_files:
    path = Path(file)
    if path.exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file}")
        missing_files.append(file)

if missing_files:
    print(f"\n  Missing files: {', '.join(missing_files)}\n")
    sys.exit(1)
else:
    print("  ✅ All required files present\n")

# Check data directory
print("[4/5] Checking data directory...")
data_dir = Path('data')
if not data_dir.exists():
    print(f"  ℹ️  data/ directory not found (will be created)")
else:
    pdf_files = list(data_dir.glob('*.pdf'))
    print(f"  Found {len(pdf_files)} PDF file(s)")
    if len(pdf_files) == 0:
        print(f"  ⚠️  No PDFs found. Add:")
        print(f"      - 10-Q4-2024-As-Filed.pdf (Apple 10-K)")
        print(f"      - tsla-20231231-gen.pdf (Tesla 10-K)")
    else:
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024**2)
            print(f"    - {pdf.name} ({size_mb:.1f} MB)")
print()

# Check Ollama
print("[5/5] Checking Ollama (Local LLM)...")
try:
    response = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/tags'],
        capture_output=True,
        timeout=2
    )
    if response.returncode == 0:
        print("  ✅ Ollama running at http://localhost:11434")
        if b'mistral' in response.stdout:
            print("  ✅ Mistral model available")
        else:
            print("  ⚠️  Mistral model not found")
            print("     Run: ollama run mistral")
    else:
        print("  ❌ Ollama not responding")
except Exception as e:
    print("  ℹ️  Ollama not found (expected if not installed)")
    print("     Visit: https://ollama.ai to install")
    print("     Then run: ollama run mistral")

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)

print("\n📝 Next steps:")
print("  1. If any packages failed, run: pip install -r requirements.txt")
print("  2. Add PDFs to data/ directory")
print("  3. Make sure Ollama is running: ollama run mistral")
print("  4. Test with: python test_runner.py")
print("  5. Or use web UI: streamlit run app.py\n")

