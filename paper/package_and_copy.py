import shutil
import subprocess
import zipfile
from pathlib import Path

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
artifact_dir = Path(r"C:\Users\Dell\.gemini\antigravity\brain\a874dfa5-dae2-4805-8cde-79a4d7ef16d5")

zip_path = paper_dir / "ConMamba_ASR_Overleaf.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(paper_dir / "main.tex", arcname="main.tex")
    z.write(paper_dir / "references.bib", arcname="references.bib")

print(f"Updated {zip_path}")

# Copy to artifact directory
shutil.copy2(paper_dir / "main.pdf", artifact_dir / "main.pdf")
shutil.copy2(zip_path, artifact_dir / "ConMamba_ASR_Overleaf.zip")
print("Copied main.pdf and zip to artifact directory.")
