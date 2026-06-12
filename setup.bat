@echo off
REM ================================================================
REM  InfraHeal AI — Setup Script (Windows)
REM  TCS & AMD AI Hackathon 2026
REM ================================================================
echo.
echo  ========================================
echo   InfraHeal AI — Environment Setup
echo  ========================================
echo.

REM 1. Create virtual environment (optional)
if not exist ".venv" (
    echo [1/4] Creating Python virtual environment...
    python -m venv .venv
) else (
    echo [1/4] Virtual environment already exists, skipping.
)

REM 2. Install dependencies
echo [2/4] Installing dependencies...
call .venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] pip install failed, trying without venv...
    pip install -r requirements.txt
)

REM 3. Generate sample data
echo [3/4] Generating synthetic infrastructure data...
python -c "from data_generator import generate_all_data; generate_all_data(save_to_disk=True); print('Done')"

REM 4. Verify setup
echo [4/4] Running verification...
python -c "
import py_compile, sys, os
sys.path.insert(0, '.')
files = ['config.py','data_generator.py','anomaly_detector.py','utils.py','gpu_tracker.py',
         'rag/knowledge_base.py','agents/base_agent.py','agents/triage_agent.py',
         'agents/rca_agent.py','agents/remediation_agent.py','agents/reporting_agent.py',
         'agents/orchestrator.py','dashboard.py','main.py']
ok = all(py_compile.compile(f, doraise=True) or print(f'  OK  {f}') or True for f in files)
print()
print('All files compile OK' if ok else 'Some files failed')
"

echo.
echo  ========================================
echo  Setup complete! Quick commands:
echo.
echo    python main.py              Launch dashboard
echo    python main.py --demo       Run CLI demo
echo    python main.py --test-llm   Test vLLM connection
echo  ========================================
echo.
