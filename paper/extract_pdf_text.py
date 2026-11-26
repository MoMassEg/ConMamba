import re
import zlib
from pathlib import Path

pdf_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.pdf")
with open(pdf_path, "rb") as f:
    data = f.read()

# Decompress all FlateDecode streams
streams = re.findall(rb"<<(.*?)/Length\s+(\d+).*?>>\s*stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
print(f"Total streams found: {len(streams)}")

text_chunks = []
for header, length, content in streams:
    if b"/FlateDecode" in header:
        try:
            decomp = zlib.decompress(content)
            # Find Tj or TJ strings
            text_matches = re.findall(rb"\((.*?)\)\s*Tj", decomp)
            if text_matches:
                page_text = b" ".join(text_matches).decode("latin1", errors="ignore")
                text_chunks.append(page_text)
        except Exception:
            pass

print(f"Text chunks found: {len(text_chunks)}")
if text_chunks:
    print("=== LAST CHUNK ===")
    print(text_chunks[-1][:500])
