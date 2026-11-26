from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
test_tex = paper_dir / "test_fig_clean.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(test_tex, "r", encoding="utf-8") as f:
    text = f.read()

probe = r"""
\path let \p1 = (c1.south), \p2 = (c2.south), \p3 = (c3.south),
          \p4 = (c1.north), \p5 = (c2.north), \p6 = (c3.north),
          \p7 = (c1.east), \p8 = (c2.west), \p9 = (c2.east), \p10 = (c3.west) in
    \pgfextra{
        \typeout{COORDS: c1_east=\x7, c2_west=\x8, gutter1=\the\dimexpr\x8-\x7\relax}
        \typeout{COORDS: c2_east=\x9, c3_west=\x10, gutter2=\the\dimexpr\x10-\x9\relax}
        \typeout{COORDS: c1_north=\y4, c2_north=\y5, c3_north=\y6}
        \typeout{COORDS: c1_south=\y1, c2_south=\y2, c3_south=\y3}
    };
"""

target = r"\end{scope}" + "\n\n" + r"\end{tikzpicture}"
text = text.replace(target, r"\end{scope}" + "\n" + probe + "\n" + r"\end{tikzpicture}")

with open(test_tex, "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(test_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "COORDS:" in line:
        print(line)
