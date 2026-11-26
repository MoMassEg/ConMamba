import re
from pathlib import Path

# Add a marker in main.tex right before \begin{thebibliography}
paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's insert \typeout{PAGE_PROBE_BIB_START} right before \begin{thebibliography}
# and \typeout{PAGE_PROBE_CONCL_START} right before \section{Conclusion}
text_probed = text.replace(r"\section{Conclusion}", r"\typeout{PAGE_PROBE_CONCL_START}" + "\n" + r"\section{Conclusion}")
text_probed = text_probed.replace(r"\begin{thebibliography}", r"\typeout{PAGE_PROBE_BIB_START}" + "\n" + r"\begin{thebibliography}")

with open(paper_dir / "main.tex", "w", encoding="utf-8") as f:
    f.write(text_probed)

import subprocess
res = subprocess.run([str(paper_dir / "tools" / "tectonic.exe"), "--print", str(paper_dir / "main.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "PAGE_PROBE" in line or ("[" in line and "]" in line and any(f"[{i}]" in line for i in range(1, 13))):
        print(line)
