#!/usr/bin/env python3
"""make_titlezero.py - zero the L (luma, low byte) plane of a standalone A8L8
sprite file, all mip levels. Byte length unchanged => valid pkg repack.

Header (18 bytes, same layout as the .pngatlas):
    u32 7 ; u32 0 ; u16 W ; u16 H ; u16 usedW ; u16 usedH ; u16 ; u16
Pixels follow at se+18. Each texel is u16 little-endian: low byte = L, high = A.
The pixel region is a full mip chain (W,H -> 2,1).

Usage:
    make_titlezero.py <in_sprites_png> <out_png>
"""
import sys, struct, os

def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dst = argv
    d = bytearray(open(src, "rb").read())
    magic, ver, flags = struct.unpack_from("<III", d, 0)
    L = d[16]
    se = 17 + L
    assert struct.unpack_from("<I", d, se)[0] == 7, hex(struct.unpack_from("<I", d, se)[0])
    w, h = struct.unpack_from("<HH", d, se + 6)
    pix_start = se + 18
    for i in range(pix_start, len(d), 2):
        d[i] = 0  # L = 0, keep A
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    open(dst, "wb").write(bytes(d))
    print("wrote %d bytes to %s (WxH=%dx%d) L plane zeroed all mips" % (len(d), dst, w, h))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))