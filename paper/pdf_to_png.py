import struct
import zlib
from pathlib import Path
import sys

sys.path.insert(0, r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\tools\py_packages")
import pypdfium2 as pdfium

def save_pdf_page_as_png(pdf_path, page_num, png_path, scale=2):
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]
    bitmap = page.render(scale=scale)
    w, h, stride = bitmap.width, bitmap.height, bitmap.stride
    buf = bitmap.buffer

    # Raw scanlines for PNG (filter type 0 followed by RGB)
    raw_scanlines = bytearray(h * (1 + w * 3))
    out_idx = 0
    for y in range(h):
        raw_scanlines[out_idx] = 0 # None filter
        out_idx += 1
        row_start = y * stride
        for x in range(w):
            px = row_start + x * 3
            raw_scanlines[out_idx] = buf[px+2]     # R
            raw_scanlines[out_idx+1] = buf[px+1]   # G
            raw_scanlines[out_idx+2] = buf[px]     # B
            out_idx += 3

    # PNG Signature
    png = bytearray(b"\x89PNG\r\n\x1a\n")

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    # IDAT chunk
    compressed = zlib.compress(raw_scanlines, 6)
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    with open(png_path, "wb") as f:
        f.write(png)
    print(f"Rendered {png_path} ({w}x{h}) successfully!")

if __name__ == "__main__":
    paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
    save_pdf_page_as_png(paper_dir / "main.pdf", 1, paper_dir / "page2.png", scale=2)
