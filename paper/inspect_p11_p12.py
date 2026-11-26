import re
import zlib
from pathlib import Path

pdf_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.pdf")
with open(pdf_path, "rb") as f:
    data = f.read()

streams = re.findall(rb"<<(.*?)/Length\s+(\d+).*?>>\s*stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)

page_texts = []
for header, length, content in streams:
    try:
        decomp = zlib.decompress(content)
        tjs = re.findall(rb"\[(.*?)\]\s*TJ", decomp)
        if tjs:
            extracted = []
            for tj in tjs:
                strs = re.findall(rb"\((.*?)\)", tj)
                extracted.append(b"".join(strs).decode("latin1", errors="ignore"))
            page_texts.append(" ".join(extracted))
    except Exception:
        pass

print(f"Total pages/streams: {len(page_texts)}")
print("=== PAGE 11 ===")
print(page_texts[-2])
print("=== PAGE 12 ===")
print(page_texts[-1])
