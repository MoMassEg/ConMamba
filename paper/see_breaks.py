import subprocess
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "main.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "[" in line and "]" in line and any(f"[{i}]" in line for i in range(1, 13)):
        print(line)
