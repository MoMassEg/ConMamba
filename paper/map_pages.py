from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(paper_dir / "temp_11p.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace all \section and \subsection with a probe printing the section name and page
import re
def repl_sec(m):
    return f"\\typeout{{PAGEMAP: page=\\thepage, sec={m.group(1)}}}\n" + m.group(0)

text_mapped = re.sub(r"(\\section\{[^}]+\})", repl_sec, text)
text_mapped = re.sub(r"(\\subsection\{[^}]+\})", repl_sec, text_mapped)
text_mapped = text_mapped.replace(r"\begin{thebibliography}", r"\typeout{PAGEMAP: page=\thepage, sec=BIBLIOGRAPHY}" + "\n" + r"\begin{thebibliography}")
text_mapped = text_mapped.replace(r"\bibitem{kingma2015adam}", r"\typeout{PAGEMAP: page=\thepage, sec=LAST_BIBITEM}" + "\n" + r"\bibitem{kingma2015adam}")

with open(paper_dir / "temp_map.tex", "w", encoding="utf-8") as f:
    f.write(text_mapped)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_map.tex")], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "PAGEMAP" in line:
        print(line)
