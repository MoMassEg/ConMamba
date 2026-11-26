import re
import zlib
from pathlib import Path

pdf_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.pdf")
with open(pdf_path, "rb") as f:
    data = f.read()

streams = re.findall(rb"<<(.*?)/Length\s+(\d+).*?>>\s*stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)

for header, length, content in streams:
    try:
        decomp = zlib.decompress(content)
        if b"Kingma" in decomp or b"Adam" in decomp or b"SpeechBrain" in decomp or b"Loshchilov" in decomp:
            print("Found bib text in stream!")
            # print characters
            printable = "".join([chr(c) if 32 <= c <= 126 else " " for c in decomp])
            print(printable[-500:])
    except Exception:
        pass
