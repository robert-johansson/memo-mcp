import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "memo"))
from memo import memo
print("memo import ok")
