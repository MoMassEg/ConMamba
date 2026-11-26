import subprocess
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

# Add page break markers or track where pages ship
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's see what text is near the end
lines = text.splitlines()
print("\n".join(lines[-40:]))
