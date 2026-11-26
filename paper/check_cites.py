import re
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Extract all \cite{...}
cites = set()
for match in re.findall(r"\\cite\{([^}]+)\}", text):
    for c in match.split(","):
        cites.add(c.strip())

# Extract all \bibitem{...}
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", text)

print(f"Total citations called in text: {len(cites)}")
print(f"Total bibitems in thebibliography: {len(bibitems)}")

uncited = [b for b in bibitems if b not in cites]
print("Bibitems not cited in text:", uncited)
