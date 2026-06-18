# experiments/text2sg/tests/conftest.py
import sys
from pathlib import Path

# Agrega experiments/ al path para que `from text2sg.x import y` funcione
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
