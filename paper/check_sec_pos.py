from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's inspect Section IV-E, IV-F, Discussion, Limitations
sec_pos = {
    "IV-E Complexity": text.find(r"\subsection{Computational Complexity"),
    "IV-F Linguistic": text.find(r"\subsection{Dialectal Phonetic"),
    "Discussion": text.find(r"\section{Discussion}"),
    "Limitations": text.find(r"\section{Limitations}"),
    "Conclusion": text.find(r"\section{Conclusion}"),
    "Bib": text.find(r"\begin{thebibliography}")
}

for k, v in sec_pos.items():
    print(f"{k}: char index {v}")
