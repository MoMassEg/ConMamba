from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
test_tex = paper_dir / "test_fig_clean.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(test_tex, "r", encoding="utf-8") as f:
    text = f.read()

probe = r"""
\path let \p1 = (p1_splits.south), \p2 = (p1_long.south), \p3 = (p2_res.south), \p4 = (p3_out.south) in
    \pgfextra{
        \typeout{NODES: p1_splits_south=\y1, p1_long_south=\y2, p2_res_south=\y3, p3_out_south=\y4}
    };
"""

text_mod = text.replace(r"\end{tikzpicture}", probe + "\n" + r"\end{tikzpicture}")
with open(test_tex, "w", encoding="utf-8") as f:
    f.write(text_mod)

res = subprocess.run([str(tectonic_exe), "--print", str(test_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "NODES:" in line:
        print(line)
