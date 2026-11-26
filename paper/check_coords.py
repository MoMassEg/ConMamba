from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tex_file = paper_dir / "test_fig.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

# Add coordinate reporting to test_fig.tex
with open(tex_file, "r", encoding="utf-8") as f:
    content = f.read()

# Add \typeout before \end{tikzpicture}
report_code = r"""
\path let \p1 = (c1.south), \p2 = (c2.south), \p3 = (c3.south),
          \p4 = (c1.north), \p5 = (c2.north), \p6 = (c3.north) in
    \pgfextra{
        \typeout{=== BOUNDING BOX REPORT ===}
        \typeout{C1: north=\y4, south=\y1}
        \typeout{C2: north=\y5, south=\y2}
        \typeout{C3: north=\y6, south=\y3}
        \typeout{===========================}
    };
"""

content_with_report = content.replace(r"\end{tikzpicture}", report_code + r"\end{tikzpicture}")

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(content_with_report)

res = subprocess.run([str(tectonic_exe), str(tex_file)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "===" in line or "C1:" in line or "C2:" in line or "C3:" in line:
        print(line)
