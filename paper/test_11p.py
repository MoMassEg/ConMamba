from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Scale Figure 1 to 0.83
text = text.replace(r"scale=0.86, every node/.style={transform shape}", r"scale=0.83, every node/.style={transform shape}")

# 2. Scale Figure 2 height to 4.4cm
text = text.replace(r"height=5.2cm,", r"height=4.4cm,")

with open(paper_dir / "temp_11p.tex", "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_11p.tex")], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
