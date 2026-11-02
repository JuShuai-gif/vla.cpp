from pathlib import Path
import subprocess,sys
d=Path(__file__).resolve().parent
# Rejected full-model candidates remain recorded; comparison picks only validated modes.
subprocess.run([sys.executable,str(d/'run_full_model_validation.py')])
subprocess.run([sys.executable,str(d/'run_comparison.py')],check=True)
