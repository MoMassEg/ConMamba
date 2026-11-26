import subprocess
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "main.tex")], capture_output=True, text=True, cwd=str(paper_dir))

# Find lines around [11] and [12]
lines = res.stdout.splitlines()
for i, line in enumerate(lines):
    if "[11]" in line or "[12]" in line:
        start = max(0, i-5)
        end = min(len(lines), i+6)
        print("\n".join(lines[start:end]))
        print("-" * 40)
