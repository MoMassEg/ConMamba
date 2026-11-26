from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Change height of fig:latency_scaling from 5.2cm to 4.6cm
text_mod = text.replace(r"height=5.2cm,", r"height=4.5cm,")

with open(paper_dir / "temp_scale_fig2.tex", "w", encoding="utf-8") as f:
    f.write(text_mod)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_scale_fig2.tex")], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
