from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Replace seamlessly
text = text.replace("extrapolate seamlessly to arbitrarily long sequences", "extrapolate directly to arbitrarily long sequences")

# Replace em-dashes
text = text.replace("domain-adaptation protocols—including parameter-efficient LoRA adaptation and unsupervised acoustic normalization—represents", "domain-adaptation protocols, including parameter-efficient LoRA adaptation and unsupervised acoustic normalization, represents")

with open(main_tex, "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed em-dashes and banned words!")

res = subprocess.run([str(tectonic_exe), str(main_tex)], capture_output=True, text=True, cwd=str(paper_dir))
print("Tectonic exit code:", res.returncode)
