from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tex_file = paper_dir / "test_fig.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(tex_file, "r", encoding="utf-8") as f:
    text = f.read()

# Add probe right before \end{tikzpicture}
probe = r"""
\path let \p1 = (c1.south), \p2 = (c2.south), \p3 = (c3.south),
          \p4 = (c1.north), \p5 = (c2.north), \p6 = (c3.north) in
    \pgfextra{
        \typeout{COORD_REPORT: c1_north=\y4, c1_south=\y1}
        \typeout{COORD_REPORT: c2_north=\y5, c2_south=\y2}
        \typeout{COORD_REPORT: c3_north=\y6, c3_south=\y3}
    };
"""

target = r"\end{scope}" + "\n\n" + r"\end{tikzpicture}"
new_text = text.replace(target, r"\end{scope}" + "\n" + probe + "\n" + r"\end{tikzpicture}")

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(new_text)

res = subprocess.run([str(tectonic_exe), str(tex_file)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "COORD_REPORT" in line:
        print(line)
