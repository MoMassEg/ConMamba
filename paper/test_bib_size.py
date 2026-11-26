from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Let's see: change \begin{thebibliography}{...} to:
# \begingroup
# \footnotesize
# \begin{thebibliography}{38}
# ...
# \end{thebibliography}
# \endgroup

text = text.replace(r"\IEEEtriggeratref{20}" + "\n", "")
old_bib = r"\begin{thebibliography}{38}"
if old_bib not in text:
    old_bib = r"\begin{thebibliography}{10}"

new_bib = r"""\begingroup
\footnotesize
\begin{thebibliography}{38}"""

text = text.replace(old_bib, new_bib)
text = text.replace(r"\end{thebibliography}", r"\end{thebibliography}" + "\n" + r"\endgroup")

with open(main_tex, "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(main_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
