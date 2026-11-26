import re
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's remove the uncited loshchilov2019adamw bibitem
old_item = r"""\bibitem{loshchilov2019adamw}
I.~Loshchilov and F.~Hutter, ``Decoupled weight decay regularization,'' in
  \emph{Proc. International Conference on Learning Representations (ICLR)},
  2019."""

if old_item in text:
    text = text.replace(old_item, "")

# Cite park2019specaugment in SpecAugment description
text = text.replace(
    r"SpecAugment~\cite{park2019specaugment}",
    r"SpecAugment~\cite{park2019specaugment}"
)
if "SpecAugment~\\cite{park2019specaugment}" not in text:
    text = text.replace(
        r"SpecAugment (4 time drops",
        r"SpecAugment~\cite{park2019specaugment} (4 time drops"
    )

with open(paper_dir / "main.tex", "w", encoding="utf-8") as f:
    f.write(text)

import subprocess
res = subprocess.run([str(paper_dir / "tools" / "tectonic.exe"), "--print", str(paper_dir / "main.tex")], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
