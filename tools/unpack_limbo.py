#!/usr/bin/env python3
"""unpack_limbo.py - extract a LIMBO .pkg into files.

    u32 count ; count x {u32 name_hash, u32 entry_off, u32 size} ; then blobs
    (entry_off is relative to the table end). Entries whose name ends in '.d'
    are stored zlib-deflated and are written out with the '.d' stripped.

Filenames come from the community Gibbed.Limbo filelists; entries with no
known name are written under __UNKNOWN/<hash>.bin. Output paths are confined
to <outdir> - a filelist entry that tries to escape it (absolute path or
'..') is refused rather than written.

Usage:
    unpack_limbo.py <pkg> <filelist_dir> <outdir>
"""
import os
import struct
import sys
import zlib


def read_filelists(directory):
    names = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".filelist"):
            continue
        with open(os.path.join(directory, fn), "rt", encoding="ascii", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                norm = line.replace("\\", "/").lower()
                names[crc32(norm)] = norm
    return names


def crc32(text):
    return zlib.crc32(text.encode("ascii")) & 0xFFFFFFFF


def safe_join(outdir, rel):
    """Join rel under outdir, refusing anything that escapes it."""
    rel = rel.replace("\\", "/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or any(":" in p for p in parts):
        raise ValueError("refusing unsafe entry name: %r" % rel)
    target = os.path.realpath(os.path.join(outdir, *parts)) if parts else os.path.realpath(outdir)
    base = os.path.realpath(outdir)
    if os.path.commonpath([base, target]) != base:
        raise ValueError("refusing entry that escapes outdir: %r" % rel)
    return target


def unpack(pkg_path, names, outdir):
    with open(pkg_path, "rb") as fh:
        raw = fh.read()
    (count,) = struct.unpack_from("<I", raw, 0)
    entries = []
    for i in range(count):
        off = 4 + i * 12
        name_hash, entry_off, size = struct.unpack_from("<III", raw, off)
        entries.append((name_hash, entry_off, size))
    base = 4 + count * 12
    named = 0
    skipped = 0
    for idx, (name_hash, entry_off, size) in enumerate(entries):
        blob = raw[base + entry_off : base + entry_off + size]
        name = names.get(name_hash)
        if name is None:
            name = "__UNKNOWN/%08X" % name_hash
        else:
            named += 1
            if name.endswith(".d"):
                inflated = inflate(blob)
                if inflated is not None:
                    blob = inflated
                    name = name[:-2]
        try:
            target = safe_join(outdir, name)
        except ValueError as exc:
            print("skip: %s" % exc, file=sys.stderr)
            skipped += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as out:
            out.write(blob)
    return count, named, skipped


def inflate(blob):
    dec = zlib.decompressobj()
    try:
        return dec.decompress(blob)
    except zlib.error:
        return None


def main(argv):
    if len(argv) != 3:
        print("usage: unpack_limbo.py <pkg> <filelist_dir> <outdir>")
        return 2
    pkg_path, filelist_dir, outdir = argv
    names = read_filelists(filelist_dir)
    total, named, skipped = unpack(pkg_path, names, outdir)
    print("entries=%d named=%d (%.0f%%) skipped=%d" % (
        total, named, 100.0 * named / total if total else 0.0, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
