from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
