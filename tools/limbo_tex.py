#!/usr/bin/env python3
"""limbo_tex.py - decode / edit / re-encode LIMBO (Playdead) texture files.

Formats handled: standalone sprites (.png/.png_a/.pngblur) and atlases (.pngatlas).
Pixel format: D3DFMT_A8L8, little-endian u16 per texel (low byte L, high byte A).
Layout:  u32 9 | u32 2 | u32 flags(1 sprite / 9 blur/ 17 atlas) | u8 len | path
         | 7 zero bytes | 11-byte tail | pixels; pixels start at path_end+18 and
         run to EOF. Atlas pixel sheets are 4096 u16 wide (virtual w/h in the
         tail at path_end+10 / path_end+12 as u16). Sprite w/h are NOT stored.

Modes:
  inspect   FILE...                  dump parsed header + geometry
  decode    FILE --w W --h H|-rect x,y,w,h --out png
  patch     FILE --out NEW  --in png --w W --h H | --rect x,y,w,h
            (in-place, same resolution; header copied verbatim -> byte-safe)
  roundtrip FILE...                  parse + re-emit, compare SHA
"""
import sys, os, struct, hashlib

USAGE = __doc__


def parse(path):
    """Return dict with header (verbatim), pixel offset, npix, geometry hints."""
    d = open(path, "rb").read()
    assert d[:4] == b"\x09\x00\x00\x00", "not a limbo texture: %s" % path
    magic, ver, flags = struct.unpack_from("<III", d, 0)
    L = d[16]
    se = 17 + L
    pix = se + 18
    assert (len(d) - pix) % 2 == 0, "pixel region not u16-aligned"
    path_s = d[17:se].decode("utf-8", "replace")
    npix = (len(d) - pix) // 2
    info = dict(file=path, size=len(d), magic=magic, ver=ver, flags=flags,
                path=path_s, pathlen=L, path_end=se, pix=pix, npix=npix)
    # virtual dims for atlases live at se+10 / se+12
    if flags == 17:
        vw, vh = struct.unpack_from("<HH", d, se + 10)
        info["virt_w"] = vw
        info["virt_h"] = vh
    return d, info


def pixels(d, info):
    return memoryview(d)[info["pix"]:]


def to_u16(buf):
    n = len(buf)
    assert n % 2 == 0
    import numpy as np
    return np.frombuffer(buf, dtype="<u2").astype(np.int64)


def sheet(px, info, width=4096):
    import numpy as np
    H = len(px) // width
    assert len(px) % width == 0 and H * width == len(px)
    return px[: width * H].reshape(H, width), H


def a8l8_to_rgba(u):
    import numpy as np
    L = (u & 0xFF).astype(np.uint8)
    A = (u >> 8).astype(np.uint8)
    return np.dstack([L, L, L, A])


def rgba_to_a8l8(arr):
    import numpy as np
    L = arr[..., 0].astype(np.int64)
    A = arr[..., 3].astype(np.int64)
    return (A << 8 | L).astype("<u2")


def cmd_inspect(files):
    for f in files:
        d, info = parse(f)
        if info["flags"] == 17:
            extra = "sheet_w=4096 virt=%dx%d" % (info.get("virt_w"), info.get("virt_h"))
        else:
            extra = ""
        print("%-58s flags=%-3d path=%s  npix=%d  %s" % (
            os.path.basename(f), info["flags"], info["path"], info["npix"], extra))


def cmd_decode(d, info, out, w=None, h=None, rect=None):
    import numpy as np
    px = to_u16(pixels(d, info))
    if rect:
        x, y, rw, rh = rect
        if info["flags"] == 17:
            M, H = sheet(px, info)
            sub = M[y : y + rh, x : x + rw]
        else:
            sub = px.reshape(y + rh, w or 4096) if False else px.reshape(rh, rw)
        arr = a8l8_to_rgba(sub)
    else:
        assert w and h and w * h == info["npix"], "bad w/h for %s (npix=%d)" % (info["file"], info["npix"])
        arr = a8l8_to_rgba(px.reshape(h, w))
    from PIL import Image
    Image.fromarray(arr, "RGBA").save(out)
    print("wrote %s" % out)


def cmd_patch(d, info, out, png, w=None, h=None, rect=None):
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(png).convert("RGBA"))
    if rect:
        x, y, rw, rh = rect
        assert img.shape[:2] == (rh, rw), "patch image %s != rect %dx%d" % (img.shape[:2], rw, rh)
        px = to_u16(pixels(d, info))
        M, H = sheet(px, info)
        M[y : y + rh, x : x + rw] = rgba_to_a8l8(img)
        new_px = M.reshape(-1).astype("<u2")
        assert len(new_px) == info["npix"]
    else:
        assert w and h and w * h == info["npix"], "bad w/h"
        assert img.shape[:2] == (h, w), "patch image %s != %dx%d" % (img.shape[:2], w, h)
        new_px = rgba_to_a8l8(img).reshape(-1)
    new = d[: info["pix"]] + new_px.tobytes()
    assert len(new) == len(d), (len(new), len(d))
    with open(out, "wb") as fh:
        fh.write(new)
    print("wrote %s (%d bytes, header verbatim, pixel region replaced)" % (out, len(new)))


def cmd_roundtrip(files):
    # Real test: re-encode the pixel region through the RGBA path and confirm the
    # re-emitted file (verbatim header + re-encoded pixels) is byte-identical.
    ok = True
    for f in files:
        d, info = parse(f)
        px = to_u16(pixels(d, info))
        back = rgba_to_a8l8(a8l8_to_rgba(px)).astype("<u2")
        reemitted = d[: info["pix"]] + back.tobytes()
        same = reemitted == d and to_u16(back.tobytes()).tolist() == px.tolist()
        ok &= same
        print("%-58s %s %s" % (os.path.basename(f), "OK" if same else "FAIL", info["path"]))
    print("ROUND TRIP:", "ALL PASS (%d)" % len(files) if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd = argv[0]
    args = argv[1:]
    if cmd == "inspect":
        return cmd_inspect(args)
    if cmd in ("decode",):
        d, info = parse(args[0])
        out = None
        w = h = None
        rect = None
        for i, a in enumerate(args[1:], 1):
            if a == "--out":
                out = args[i + 1]
            elif a == "--w":
                w = int(args[i + 1])
            elif a == "--h":
                h = int(args[i + 1])
            elif a == "--rect":
                rect = tuple(int(v) for v in args[i + 1].split(","))
        assert out
        return cmd_decode(d, info, out, w, h, rect)
    if cmd == "patch":
        d, info = parse(args[0])
        out = png = None
        w = h = None
        rect = None
        for i, a in enumerate(args[1:], 1):
            if a == "--out":
                out = args[i + 1]
            elif a == "--in":
                png = args[i + 1]
            elif a == "--w":
                w = int(args[i + 1])
            elif a == "--h":
                h = int(args[i + 1])
            elif a == "--rect":
                rect = tuple(int(v) for v in args[i + 1].split(","))
        assert out and png
        return cmd_patch(d, info, out, png, w, h, rect)
    if cmd == "roundtrip":
        return cmd_roundtrip(args)
    print("unknown command: %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))