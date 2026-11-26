import re
import zlib
from pathlib import Path

pdf_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.pdf")
with open(pdf_path, "rb") as f:
    data = f.read()

streams = re.findall(rb"<<(.*?)/Length\s+(\d+).*?>>\s*stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)

page_texts = []
for header, length, content in streams:
    if b"/Type\s*/Page" in header or True:
        try:
            decomp = zlib.decompress(content)
            # Find TJ operators
            tjs = re.findall(rb"\[(.*?)\]\s*TJ", decomp)
            if tjs:
                extracted = []
                for tj in tjs:
                    strs = re.findall(rb"\((.*?)\)", tj)
                    extracted.append(b"".join(strs).decode("latin1", errors="ignore"))
                page_texts.append(" ".join(extracted))
        except Exception:
            pass

print(f"Streams with text: {len(page_texts)}")
for i, pt in enumerate(page_texts):
    print(f"Stream {i} length: {len(pt)}, preview: {pt[:100]}")
