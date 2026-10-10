#!/usr/bin/env python3
"""limbo_workflow.py - extract LIMBO textures to editable PNGs, then pack edits back.

A thin orchestrator over `limbo_tex.py` (the format engine) that gives the
human-edits-art workflow:

    # 1. dump every sprite + atlas member to clean RGBA PNGs (L as gray, A as alpha)
    python limbo_workflow.py extract --boot lab\\boot --out assets

    # 2. hand the assets/ tree to a human, they edit PNGs in any image editor

    # 3. rebuild only the textures whose PNG changed, and repack the pkg
    python limbo_workflow.py pack --boot lab\\boot --assets assets ^
        --pkg-in pristine\\Limbo\\limbo_boot.pkg --filelists <filelist_dir> ^
        --pkg-out lab\\renders\\limbo_boot_edited.pkg

`assets/MANIFEST.json` records, for each PNG, its source texture and (for atlas
members) the manifest rect + resource path, plus the PNG's original sha256 so
`pack` can tell which assets the human actually changed.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import limbo_tex as T  # noqa: E402
from PIL import Image  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest_rows(manifest_path):
    """Yield (x, y, w, h, off_x, off_y, path) from an atlas .txt manifest."""
    with open(manifest_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            # TAB-separated: x y w h offX offY "path"
            left, _, quoted = line.partition("\t")
            parts = left.split()
            if len(parts) != 6:
                parts = line.split("\t")
                quoted = parts[-1]
                parts = parts[:6]
            try:
                x, y, w, h, ox, oy = (int(v) for v in parts)
            except ValueError:
                continue
            path = quoted.strip().strip('"')
            yield x, y, w, h, ox, oy, path


def extract(boot, out):
    if os.path.exists(out):
        print("extract: %s already exists; delete it first" % out)
        return 2
    os.makedirs(out)
    manifest = {"boot": os.path.abspath(boot), "assets": []}
    # --- standalone sprites ---
    sprite_dir = os.path.join(boot, "derived", "pc", "data")
    n_spr = 0
    for root, _dirs, files in os.walk(sprite_dir):
        for fn in files:
            if not (fn.endswith(".png") or fn.endswith(".png_a") or fn.endswith(".pngblur")):
                continue
            src = os.path.join(root, fn)
            try:
                d, info = T.parse(src)
            except Exception:
                continue
            if info["flags"] == 17:
                continue
            rel = os.path.relpath(src, boot)
            png_rel = os.path.join(rel + ".png") if not rel.endswith(".png") else rel
            png_path = os.path.join(out, png_rel)
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            _decode_to(d, info, info["img_w"], info["img_h"], None, png_path)
            manifest["assets"].append({
                "kind": "sprite",
                "texture": rel.replace(os.sep, "/"),
                "png": png_rel.replace(os.sep, "/"),
                "img_w": info["img_w"], "img_h": info["img_h"],
                "sha": sha(png_path),
            })
            n_spr += 1
    # --- atlas members ---
    atlas_dir = os.path.join(boot, "derived", "pc", "data", "texture", "atlas")
    n_atl = 0
    if os.path.isdir(atlas_dir):
        for fn in sorted(os.listdir(atlas_dir)):
            if not fn.endswith(".pngatlas"):
                continue
            src = os.path.join(atlas_dir, fn)
            d, info = T.parse(src)
            manifest_path = os.path.join(boot, "data", "texture", "atlas", fn[:-len("pngatlas")] + "txt")
            if not os.path.exists(manifest_path):
                continue
            name = fn[:-len(".pngatlas")]
            px = T.to_u16(T.pixels(d, info))
            for (x, y, w, h, ox, oy, path) in read_manifest_rows(manifest_path):
                sub = px[: info["img_h"] * info["img_w"]].reshape(info["img_h"], info["img_w"])[y:y + h, x:x + w]
                if sub.size == 0:
                    continue
                png_rel = "%s/%s.png" % (name, path.replace("\\", "/"))
                png_path = os.path.join(out, png_rel)
                os.makedirs(os.path.dirname(png_path), exist_ok=True)
                Image.fromarray(T.a8l8_to_rgba(sub), "RGBA").save(png_path)
                manifest["assets"].append({
                    "kind": "atlas",
                    "texture": os.path.relpath(src, boot).replace(os.sep, "/"),
                    "png": png_rel.replace(os.sep, "/"),
                    "rect": [x, y, w, h], "path": path,
                    "img_w": info["img_w"], "img_h": info["img_h"],
                    "sha": sha(png_path),
                })
                n_atl += 1
    with open(os.path.join(out, "MANIFEST.json"), "wt", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("extracted %d standalone sprites + %d atlas members -> %s" % (n_spr, n_atl, out))
    return 0


def _decode_to(d, info, w, h, rect, png_path):
    px = T.to_u16(T.pixels(d, info))
    if rect:
        x, y, rw, rh = rect
        sub = px[: info["img_h"] * info["img_w"]].reshape(info["img_h"], info["img_w"])[y:y + rh, x:x + rw]
        arr = T.a8l8_to_rgba(sub)
    else:
        arr = T.a8l8_to_rgba(px[: w * h].reshape(h, w))
    Image.fromarray(arr, "RGBA").save(png_path)


def pack(boot, assets, pkg_in, filelists, pkg_out):
    manifest = json.load(open(os.path.join(assets, "MANIFEST.json"), "rt", encoding="utf-8"))
    # group edits per source texture
    edited_sprites = {}
    edited_atlas = {}
    for a in manifest["assets"]:
        png_path = os.path.join(assets, a["png"])
        if not os.path.exists(png_path) or sha(png_path) == a["sha"]:
            continue
        if a["kind"] == "sprite":
            edited_sprites[a["texture"]] = png_path
        else:
            edited_atlas.setdefault(a["texture"], []).append((a["rect"], png_path))
    changed = {}  # texture rel -> new bytes
    for tex, png in edited_sprites.items():
        d, info = T.parse(os.path.join(boot, tex))
        src = Image.open(png).convert("RGBA")
        assert src.size == (info["img_w"], info["img_h"]), "%s != %dx%d" % (png, info["img_w"], info["img_h"])
        px = T.build_mips(src)
        assert len(px) == 2 * info["npix"]
        changed[tex] = d[: info["pix"]] + px
    for tex, members in edited_atlas.items():
        d, info = T.parse(os.path.join(boot, tex))
        px = T.to_u16(T.pixels(d, info)).copy()
        for rect, png in members:
            x, y, w, h = rect
            src = Image.open(png).convert("RGBA")
            assert src.size == (w, h), "%s != %dx%d" % (png, w, h)
            for off, k, Wk, Hk in T.mip_offsets(len(px), info["img_w"], info["img_h"]):
                xk, yk = min(x >> k, Wk - 1), min(y >> k, Hk - 1)
                wk, hk = min(max(1, w >> k), Wk - xk), min(max(1, h >> k), Hk - yk)
                sub = src.resize((wk, hk), Image.BOX) if (wk, hk) != (w, h) else src
                cur = px[off:off + Wk * Hk].reshape(Hk, Wk)
                cur[yk:yk + hk, xk:xk + wk] = T.rgba_to_a8l8(np.asarray(sub))
        changed[tex] = d[: info["pix"]] + px.astype("<u2").tobytes()
    if not changed:
        print("pack: no edited PNGs found (all match MANIFEST shas); nothing to do")
        return 1
    # write the rebuilt textures to a temp dir for repack
    tmp = os.path.join(assets, "_packed")
    os.makedirs(tmp, exist_ok=True)
    argv = [pkg_in, filelists, pkg_out, "--force"]
    for tex, data in changed.items():
        out_tex = os.path.join(tmp, os.path.basename(tex))
        with open(out_tex, "wb") as fh:
            fh.write(data)
        entry = tex + ".d"
        argv += ["--entry", entry, "--in", out_tex]
        print("  %-56s %s" % (tex, entry))
    import repack_limbo as R
    rc = R.main(argv)
    if rc == 0:
        print("packed %d changed texture(s) into %s" % (len(changed), pkg_out))
    return rc


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    ap = argparse.ArgumentParser(add_help=False)
    sub = argv[0]
    rest = argv[1:]
    if sub == "extract":
        ap.add_argument("--boot", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(rest)
        return extract(a.boot, a.out)
    if sub == "pack":
        ap.add_argument("--boot", required=True)
        ap.add_argument("--assets", required=True)
        ap.add_argument("--pkg-in", dest="pkg_in", required=True)
        ap.add_argument("--filelists", required=True)
        ap.add_argument("--pkg-out", dest="pkg_out", required=True)
        a = ap.parse_args(rest)
        return pack(a.boot, a.assets, a.pkg_in, a.filelists, a.pkg_out)
    print("unknown command: %s" % sub, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
