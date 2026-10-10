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
  extract   FILE --out png [--rect x,y,w,h]
            Decode to PNG. Standalone sprites use their stored W/H; atlases
            decode level 0 (4096 wide) or a manifest rect.
  decode    FILE --w W --h H|--rect x,y,w,h --out png   (explicit dims)
  pack      FILE --out NEW --in png
            Rebuild a texture from a level-0 PNG: the full mip chain is
            regenerated (box downsample) and the header copied verbatim, so the
            byte length is unchanged. This is the write side of `extract`.
  patch     FILE --out NEW  --in png --w W --h H | --rect x,y,w,h [--mips]
            (in-place, same resolution; header copied verbatim -> byte-safe.
            --mips, for atlas rects, also writes the downscaled edit into every
            mip level so small/blur samples don't show stale art.)
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
    img_w, img_h = struct.unpack_from("<HH", d, se + 6)
    info = dict(file=path, size=len(d), magic=magic, ver=ver, flags=flags,
                path=path_s, pathlen=L, path_end=se, pix=pix, npix=npix,
                img_w=img_w, img_h=img_h)
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


def mip_dims(W, H):
    """Mip chain dimensions, level 0 down to the level where a side reaches 1."""
    dims = []
    w, h = W, H
    while True:
        dims.append((w, h))
        if w == 1 or h == 1:
            break
        w, h = max(1, w // 2), max(1, h // 2)
    return dims


def mip_offsets(px_len, W, H):
    """Yield (offset, level, Wk, Hk) for a concatenated mip chain."""
    off = 0
    for k, (w, h) in enumerate(mip_dims(W, H)):
        if off + w * h > px_len:
            break
        yield off, k, w, h
        off += w * h


def build_mips(img):
    """Return the concatenated A8L8 mip chain for a level-0 RGBA image.

    The engine's mips are 2x2 box averages with FLOOR, computed independently on
    the L and A bytes (verified byte-exact against the shipped title sprite), not
    a PIL BOX resize.
    """
    import numpy as np
    arr = np.asarray(img.convert("RGBA"))
    L = arr[..., 0].astype(np.uint16)
    A = arr[..., 3].astype(np.uint16)
    h, w = L.shape
    assert w == (w & -w) and h == (h & -h), "level 0 must be power-of-two (%dx%d)" % (w, h)
    parts = [((A << 8) | L).astype("<u2").tobytes()]
    while w > 1 and h > 1:
        L = L.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3)) >> 2
        A = A.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3)) >> 2
        w, h = w // 2, h // 2
        parts.append(((A << 8) | L).astype("<u2").tobytes())
    return b"".join(parts)


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
        if not (w and h):
            w, h = info["img_w"], info["img_h"]
        assert w * h <= info["npix"], "bad w/h for %s (npix=%d)" % (info["file"], info["npix"])
        arr = a8l8_to_rgba(px[: w * h].reshape(h, w))
    from PIL import Image
    Image.fromarray(arr, "RGBA").save(out)
    print("wrote %s (%dx%d)" % (out, arr.shape[1], arr.shape[0]))


def cmd_patch(d, info, out, png, w=None, h=None, rect=None, mips=False):
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(png).convert("RGBA"))
    if rect:
        x, y, rw, rh = rect
        assert img.shape[:2] == (rh, rw), "patch image %s != rect %dx%d" % (img.shape[:2], rw, rh)
        if mips:
            assert info["flags"] == 17, "--mips is for atlas rects"
            px = to_u16(pixels(d, info)).copy()
            src = Image.open(png).convert("RGBA")
            for off, k, Wk, Hk in mip_offsets(len(px), info["img_w"], info["img_h"]):
                xk, yk = min(x >> k, Wk - 1), min(y >> k, Hk - 1)
                wk, hk = min(max(1, rw >> k), Wk - xk), min(max(1, rh >> k), Hk - yk)
                sub = np.asarray(src if src.size == (wk, hk) else src.resize((wk, hk), Image.BOX))
                cur = px[off : off + Wk * Hk].reshape(Hk, Wk)
                cur[yk : yk + hk, xk : xk + wk] = rgba_to_a8l8(sub)
            new_px = px.astype("<u2")
        else:
            px = to_u16(pixels(d, info))
            M, H = sheet(px, info)
            M[y : y + rh, x : x + rw] = rgba_to_a8l8(img)
            new_px = M.reshape(-1).astype("<u2")
        assert len(new_px) == info["npix"]
    else:
        assert w and h, "bad w/h"
        assert img.shape[:2] == (h, w), "patch image %s != %dx%d" % (img.shape[:2], w, h)
        # rebuild the whole mip chain from the edited level 0
        import PIL.Image as PILImage
        new_px = np.frombuffer(build_mips(PILImage.fromarray(img, "RGBA")), dtype="<u2")
        assert len(new_px) == info["npix"], (len(new_px), info["npix"])
    new = d[: info["pix"]] + new_px.tobytes()
    assert len(new) == len(d), (len(new), len(d))
    with open(out, "wb") as fh:
        fh.write(new)
    print("wrote %s (%d bytes, header verbatim, pixel region replaced%s)" % (
        out, len(new), ", all mip levels" if (rect and mips) else ""))


def cmd_pack(d, info, out, png):
    """Rebuild a texture from a level-0 PNG: regenerate the mip chain, keep header."""
    from PIL import Image
    src = Image.open(png).convert("RGBA")
    assert src.size == (info["img_w"], info["img_h"]), (
        "PNG %s != %dx%d" % (src.size, info["img_w"], info["img_h"]))
    px = build_mips(src)
    assert len(px) == 2 * info["npix"], (len(px), 2 * info["npix"])
    new = d[: info["pix"]] + px
    assert len(new) == len(d), (len(new), len(d))
    with open(out, "wb") as fh:
        fh.write(new)
    print("wrote %s (%d bytes, %dx%d + mips)" % (out, len(new), info["img_w"], info["img_h"]))


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
    if cmd in ("decode", "extract"):
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
    if cmd == "pack":
        d, info = parse(args[0])
        out = png = None
        for i, a in enumerate(args[1:], 1):
            if a == "--out":
                out = args[i + 1]
            elif a == "--in":
                png = args[i + 1]
        assert out and png
        return cmd_pack(d, info, out, png)
    if cmd == "patch":
        d, info = parse(args[0])
        out = png = None
        w = h = None
        rect = None
        mips = "--mips" in args
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
        return cmd_patch(d, info, out, png, w, h, rect, mips)
    if cmd == "roundtrip":
        return cmd_roundtrip(args)
    print("unknown command: %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))