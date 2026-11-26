import struct
import zlib
from pathlib import Path

def crop_png(input_png, output_png, crop_box):
    # crop_box = (x0, y0, x1, y1)
    with open(input_png, "rb") as f:
        data = f.read()

    # Read IHDR
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]

    # Decompress IDAT chunks
    idat = bytearray()
    idx = 8
    while idx < len(data):
        length = struct.unpack(">I", data[idx:idx+4])[0]
        chunk_type = data[idx+4:idx+8]
        if chunk_type == b"IDAT":
            idat += data[idx+8:idx+8+length]
        idx += 12 + length

    decomp = zlib.decompress(idat)

    x0, y0, x1, y1 = crop_box
    new_w = x1 - x0
    new_h = y1 - y0

    stride = 1 + w * 3
    new_stride = 1 + new_w * 3
    new_raw = bytearray(new_h * new_stride)

    out_row = 0
    for y in range(y0, y1):
        row_start = y * stride
        new_row_start = out_row * new_stride
        new_raw[new_row_start] = 0 # filter type none
        # copy pixels from x0 to x1
        src_start = row_start + 1 + x0 * 3
        src_end = row_start + 1 + x1 * 3
        dst_start = new_row_start + 1
        dst_end = dst_start + new_w * 3
        new_raw[dst_start:dst_end] = decomp[src_start:src_end]
        out_row += 1

    # Encode new PNG
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr_data = struct.pack(">IIBBBBB", new_w, new_h, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    compressed = zlib.compress(new_raw, 6)
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

    iend_crc = zlib.crc32(b"IEND")
    png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    with open(output_png, "wb") as f:
        f.write(png)
    print(f"Saved cropped {output_png} ({new_w}x{new_h})")

if __name__ == "__main__":
    paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
    crop_png(paper_dir / "main_page2.png", paper_dir / "figure1_preview.png", (40, 50, 1184, 880))
    artifact_dir = Path(r"C:\Users\Dell\.gemini\antigravity\brain\a874dfa5-dae2-4805-8cde-79a4d7ef16d5")
    crop_png(paper_dir / "main_page2.png", artifact_dir / "figure1_preview.png", (40, 50, 1184, 880))
