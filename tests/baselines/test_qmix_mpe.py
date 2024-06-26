import subprocess
import sys
import os

def run_script(script_path, *args):
    result = subprocess.run([sys.executable, script_path, *args], capture_output=True, text=True)
    return result

def test_script_with_arguments():
    script_path = os.path.join('baselines/QLearning/qmix.py')
    result = run_script(script_path, '+alg=qmix_mpe', '+env=mpe_spread','alg.TOTAL_TIMESTEPS=1e4', 'WANDB_MODE=disabled')
    assert result.returncode == 0


test_script_with_arguments()