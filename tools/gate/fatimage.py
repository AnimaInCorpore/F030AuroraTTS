#!/usr/bin/env python3
"""A 1.44 MB FAT12 floppy image (Atari-readable geometry: 512-byte sectors,
18 a track, 2 sides, media $F0) from a host folder, and reading files back.

  fatimage.py build FOLDER IMAGE
  fatimage.py get IMAGE PATH OUT        (PATH like AURORA/MINT.LOG)

Names must already be 8.3; directories nest to any depth.
"""
import struct
import sys
from pathlib import Path

SECTORS, SPT, HEADS, SPF, ROOT_ENTRIES = 2880, 18, 2, 9, 224
FAT1 = 1
ROOT = FAT1 + 2 * SPF                     # sector 19
DATA = ROOT + ROOT_ENTRIES * 32 // 512    # sector 33: cluster 2
CLUSTERS = SECTORS - DATA


def name83(name):
    base, _, ext = name.upper().partition(".")
    if len(base) > 8 or len(ext) > 3 or not base:
        raise SystemExit(f"not an 8.3 name: {name}")
    return base.ljust(8).encode() + ext.ljust(3).encode()


class Image:
    def __init__(self):
        self.data = bytearray(SECTORS * 512)
        self.fat = [0] * (CLUSTERS + 2)
        self.fat[0], self.fat[1] = 0xFF0, 0xFFF
        self.next = 2

    def alloc(self, payload):
        """Clusters for payload (one sector each); returns the first, or 0."""
        if not payload:
            return 0
        n = (len(payload) + 511) // 512
        first = self.next
        if first + n > CLUSTERS + 2:
            raise SystemExit("the image is full")
        for i in range(n):
            c = first + i
            self.fat[c] = c + 1 if i + 1 < n else 0xFFF
            off = (DATA + c - 2) * 512
            chunk = payload[i * 512:(i + 1) * 512]
            self.data[off:off + len(chunk)] = chunk
        self.next += n
        return first

    @staticmethod
    def entry(name, attr, cluster, size):
        return name83(name) + bytes([attr]) + bytes(10) + struct.pack("<HHHI", 0, 0x5000, cluster, size)

    def directory(self, folder, parent_cluster):
        """The entries of a subdirectory's contents (without . and ..)."""
        out = b""
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                out += self.entry(p.name, 0x10, self.subdir(p, parent_cluster), 0)
            else:
                body = p.read_bytes()
                out += self.entry(p.name, 0x00, self.alloc(body), len(body))
        return out

    def subdir(self, folder, parent_cluster):
        # reserve the directory's clusters first, so . can name itself
        n_entries = 2 + sum(1 for _ in folder.iterdir())
        size = ((n_entries * 32 + 511) // 512) * 512
        me = self.alloc(bytes(size))
        body = (b".          " + bytes([0x10]) + bytes(10) + struct.pack("<HHHI", 0, 0x5000, me, 0)
                + b"..         " + bytes([0x10]) + bytes(10) + struct.pack("<HHHI", 0, 0x5000, parent_cluster, 0))
        body += self.directory(folder, me)
        c, i = me, 0
        while True:
            off = (DATA + c - 2) * 512
            self.data[off:off + 512] = body[i * 512:(i + 1) * 512].ljust(512, b"\0")
            i += 1
            if self.fat[c] >= 0xFF8:
                break
            c = self.fat[c]
        return me

    def build(self, folder):
        boot = bytearray(512)
        boot[0:2] = b"\x60\x38"
        boot[2:8] = b"AURORA"
        boot[8:11] = b"\x12\x34\x56"
        struct.pack_into("<HBHBHHBHHH", boot, 11, 512, 1, 1, 2, ROOT_ENTRIES, SECTORS, 0xF0, SPF, SPT, HEADS)
        self.data[0:512] = boot
        root = self.directory(folder, 0)
        if len(root) > ROOT_ENTRIES * 32:
            raise SystemExit("too many root entries")
        self.data[ROOT * 512:ROOT * 512 + len(root)] = root
        fat = bytearray(SPF * 512)
        for i in range(0, len(self.fat) - 1, 2):
            a, b = self.fat[i], self.fat[i + 1] if i + 1 < len(self.fat) else 0
            fat[i * 3 // 2:i * 3 // 2 + 3] = bytes([a & 0xFF, ((a >> 8) & 0xF) | ((b & 0xF) << 4), (b >> 4) & 0xFF])
        for k in range(2):
            off = (FAT1 + k * SPF) * 512
            self.data[off:off + len(fat)] = fat
        return bytes(self.data)


def read_fat(img):
    raw = img[FAT1 * 512:(FAT1 + SPF) * 512]

    def get(c):
        v = raw[c * 3 // 2] | (raw[c * 3 // 2 + 1] << 8)
        return (v >> 4) if c & 1 else (v & 0xFFF)
    return get


def chain(img, first):
    get = read_fat(img)
    c = first
    while 2 <= c < 0xFF8:
        yield c
        c = get(c)


def entries(img, cluster):
    if cluster == 0:
        blob = img[ROOT * 512:DATA * 512]
    else:
        blob = b"".join(img[(DATA + c - 2) * 512:(DATA + c - 1) * 512] for c in chain(img, cluster))
    for i in range(0, len(blob), 32):
        e = blob[i:i + 32]
        if e[0] == 0:
            break
        if e[0] == 0xE5 or e[11] == 0x0F:
            continue
        name = e[0:8].decode(errors="replace").rstrip()
        ext = e[8:11].decode(errors="replace").rstrip()
        cl, size = struct.unpack("<HI", e[26:32])
        yield (name + ("." + ext if ext else "")), e[11], cl, size


def get_file(img, path):
    cluster = 0
    parts = path.upper().split("/")
    for i, part in enumerate(parts):
        for name, attr, cl, size in entries(img, cluster):
            if name == part:
                if i + 1 == len(parts):
                    blob = b"".join(img[(DATA + c - 2) * 512:(DATA + c - 1) * 512] for c in chain(img, cl))
                    return blob[:size]
                cluster = cl
                break
        else:
            return None
    return None


if __name__ == "__main__":
    if sys.argv[1] == "build":
        Path(sys.argv[3]).write_bytes(Image().build(Path(sys.argv[2])))
    else:
        data = get_file(Path(sys.argv[2]).read_bytes(), sys.argv[3])
        if data is None:
            raise SystemExit("not found")
        Path(sys.argv[4]).write_bytes(data)
