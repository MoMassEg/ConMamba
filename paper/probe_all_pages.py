from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Insert probes before sections and before bibitems
text_probe = text
text_probe = text_probe.replace(r"\section{Experiments}", r"\typeout{PAGE PROBE: Exp is on page \thepage}" + "\n" + r"\section{Experiments}")
text_probe = text_probe.replace(r"\section{Discussion}", r"\typeout{PAGE PROBE: Disc is on page \thepage}" + "\n" + r"\section{Discussion}")
text_probe = text_probe.replace(r"\section{Limitations}", r"\typeout{PAGE PROBE: Limits is on page \thepage}" + "\n" + r"\section{Limitations}")
text_probe = text_probe.replace(r"\section{Conclusion}", r"\typeout{PAGE PROBE: Concl is on page \thepage}" + "\n" + r"\section{Conclusion}")
text_probe = text_probe.replace(r"\begin{thebibliography}", r"\typeout{PAGE PROBE: BibStart is on page \thepage}" + "\n" + r"\begin{thebibliography}")
text_probe = text_probe.replace(r"\bibitem{kingma2015adam}", r"\typeout{PAGE PROBE: Kingma is on page \thepage}" + "\n" + r"\bibitem{kingma2015adam}")

with open(paper_dir / "temp_probe.tex", "w", encoding="utf-8") as f:
    f.write(text_probe)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_probe.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "PAGE PROBE" in line:
        print(line)
