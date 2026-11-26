from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Add \setlength{\itemsep}{0.5pt plus 0.5pt minus 0.2pt} right after \begin{thebibliography}{10}
old_bib = r"\begin{thebibliography}{10}"
new_bib = r"""\begin{thebibliography}{10}
\setlength{\itemsep}{0.8pt plus 0.4pt minus 0.2pt}
\setlength{\parskip}{0pt}"""

text_mod = text.replace(old_bib, new_bib)

with open(main_tex, "w", encoding="utf-8") as f:
    f.write(text_mod)

res = subprocess.run([str(tectonic_exe), "--print", str(main_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
