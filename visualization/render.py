import os
import struct
import zlib
import binascii
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "slime_console.txt"
OUTPUT = ROOT / "outputs" / "slime_pattern.png"

SHADES = " .:-=+*#%@"

PALETTE = [
    (2, 5, 18),
    (4, 20, 40),
    (0, 55, 85),
    (0, 105, 130),
    (0, 155, 145),
    (35, 195, 115),
    (125, 225, 65),
    (210, 235, 45),
    (255, 180, 35),
    (255, 245, 190)
]

SCALE = 6


def png_chunk(name, data):
    crc = binascii.crc32(name + data) & 0xffffffff
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", crc)
    )


def main():
    if not INPUT.exists():
        print("ERROR: slime_console.txt not found!")
        print("Create it inside the outputs folder.")
        return

    text = INPUT.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start = None

    for i, line in enumerate(lines):
        if "FINAL SLIME PATTERN" in line:
            start = i + 1
            break

    if start is None:
        print("ERROR: FINAL SLIME PATTERN not found.")
        print("Copy the complete LeetGPU console output.")
        return

    rows = []

    for line in lines[start:]:
        if line.strip() and set(line.strip()) == {"="}:
            break

        line = line.rstrip("\r")

        if line and all(c in SHADES for c in line):
            rows.append(line)

    if not rows:
        print("ERROR: No slime pattern rows found.")
        return

    width = max(len(row) for row in rows)

    rows = [row.ljust(width) for row in rows]

    raw_image = bytearray()

    for row in rows:
        pixels = bytearray()

        for char in row:
            level = SHADES.index(char)
            color = bytes(PALETTE[level])
            pixels.extend(color * SCALE)

        for _ in range(SCALE):
            raw_image.extend(b"\x00" + pixels)

    image_width = width * SCALE
    image_height = len(rows) * SCALE

    header = struct.pack(
        ">IIBBBBB",
        image_width,
        image_height,
        8,
        2,
        0,
        0,
        0
    )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw_image, 9))
        + png_chunk(b"IEND", b"")
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(png)

    print("SLIME VISUALIZATION GENERATED!")
    print("Image:", OUTPUT)
    print("Dimensions:", image_width, "x", image_height)
    print("Pattern rows:", len(rows))


if __name__ == "__main__":
    main()