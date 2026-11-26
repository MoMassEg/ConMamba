from pathlib import Path
import re
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Find all \bibitem{...}
items = re.findall(r"\\bibitem\{([^}]+)\}", text)
print(f"Total bibitems: {len(items)}")

# Put a probe at every 5th bibitem
probe_text = text
for i, item in enumerate(items):
    if i % 3 == 0 or i == len(items) - 1:
        probe_text = probe_text.replace(f"\\bibitem{{{item}}}", f"\\typeout{{BIB_PROBE: #{i+1} {item} is on page \\thepage}}\n\\bibitem{{{item}}}")

with open(paper_dir / "temp_bib_map.tex", "w", encoding="utf-8") as f:
    f.write(probe_text)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_bib_map.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "BIB_PROBE:" in line:
        print(line)
