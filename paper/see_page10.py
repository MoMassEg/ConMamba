from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Find text between \section{Discussion} and \section{Conclusion}
disc_idx = text.find(r"\section{Discussion}")
concl_idx = text.find(r"\section{Conclusion}")

print(text[disc_idx:concl_idx])
