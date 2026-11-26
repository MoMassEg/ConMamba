from pathlib import Path

# Check where references end in main.tex
paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
# Find \begin{thebibliography}
for i, line in enumerate(lines):
    if r"\begin{thebibliography}" in line:
        print(f"thebibliography starts at line {i+1}")
        break
