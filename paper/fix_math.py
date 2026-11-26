from pathlib import Path

tex_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.tex")
text = tex_path.read_text(encoding="utf-8")

# Fix ($p = 0.018$)
text = text.replace("( = 0.018$)", "($p = 0.018$)")

tex_path.write_text(text, encoding="utf-8")
print("Fixed math mode cleanly in fix_math.py")
