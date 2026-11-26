import re
from pathlib import Path

pdf_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.pdf")

with open(pdf_path, "rb") as f:
    data = f.read()

# Look for /Count in pages object
matches = re.findall(rb"/Type\s*/Pages.*?/Count\s+(\d+)", data, re.DOTALL)
if matches:
    print("Page counts found:", [int(m) for m in matches])
    print("Final page count:", max([int(m) for m in matches]))
else:
    # Look for /Page objects
    page_objs = re.findall(rb"/Type\s*/Page[^s]", data)
    print("Page objects found:", len(page_objs))
