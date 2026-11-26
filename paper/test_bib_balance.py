from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Let's see: if we use \small or \footnotesize or \balance
# Or check how many lines we can save
# In Section IV, Table III, Table IV, Table V
# Let's see what happens if we use \IEEEtriggeratref{20}
test_text = text.replace(r"\begin{thebibliography}{10}", r"\IEEEtriggeratref{20}" + "\n" + r"\begin{thebibliography}{38}")

with open(main_tex, "w", encoding="utf-8") as f:
    f.write(test_text)

res = subprocess.run([str(tectonic_exe), "--print", str(main_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
