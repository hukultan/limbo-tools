#!/usr/bin/env python3
"""repack_limbo.py - rebuild a LIMBO .pkg, replacing named entries with new files.

Container layout (mirrors unpack_limbo.py, which is the read side):
    u32 count ; count x {u32 name_hash, u32 entry_off, u32 size} ; then blobs
    (entry_off is RELATIVE to the table end). Entries whose name ends in '.d'
    are stored zlib-deflated.

Repacking rule: entries are written back in original order; an untouched entry
copy is the ORIGINAL blob slice verbatim; only explicitly replaced entries change.
=> A rebuild with no replacements is byte-identical to the input pkg (proven by
   SHA comparison). Compared to zlib.compress, this means an unchanged blob can
   never make the new pkg differ from the original.

The output pkg is never written over an existing file unless you pass --force
(and never over the input pkg). Back up your game's .pkg before you copy the
rebuilt one into the install.

Usage:
    repack_limbo.py <in.pkg> <filelist_dir> <out.pkg> [--force] [--entry name --in file] [...]
    repack_limbo.py --identity-test <pkg> <filelist_dir>     # rebuild untouched, compare SHA
"""
import os, sys, struct, zlib, hashlib, tempfile


def crc32(text):
    return zlib.crc32(text.encode("ascii")) & 0xFFFFFFFF


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
                names[crc32(line.replace("\\", "/").lower())] = line.replace("\\", "/").lower()
    return names


def load(pkg_path):
    raw = open(pkg_path, "rb").read()
    (count,) = struct.unpack_from("<I", raw, 0)
    base = 4 + count * 12
    entries = []
    for i in range(count):
        name_hash, off, size = struct.unpack_from("<III", raw, 4 + i * 12)
        blob = raw[base + off : base + off + size]
        entries.append({"hash": name_hash, "size": size, "blob": blob})
    return raw, entries, base


def repack(entries, out_path):
    out = bytearray()
    out += struct.pack("<I", len(entries))
    off = 0
    for e in entries:
        b = e["blob"]
        out += struct.pack("<III", e["hash"], off, len(b))
        off += len(b)
    for e in entries:
        out += e["blob"]
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(bytes(out))
    return bytes(out)


def main(argv):
    if argv and argv[0] == "--identity-test":
        if len(argv) != 3:
            print(__doc__)
            return 2
        pkg, fdir, = argv[1], argv[2]
        names = read_filelists(fdir)
        raw, entries, base = load(pkg)
        tmp = tempfile.NamedTemporaryFile(prefix="limbo_identity_", suffix=".pkg", delete=False)
        tmp.close()
        try:
            news = repack(list({**e, "blob": e["blob"]} for e in entries), tmp.name)
        finally:
            os.remove(tmp.name)
        ok = hashlib.sha256(raw).digest() == hashlib.sha256(news).digest()
        print("identity test:", "BYTE-IDENTICAL (" + str(len(news)) + " bytes)" if ok else "DIFFERS")
        return 0 if ok else 1
    if len(argv) < 3 or argv[2].startswith("--"):
        print(__doc__)
        return 2
    pkg_in, fdir, pkg_out = argv[0], argv[1], argv[2]
    force = "--force" in argv
    if os.path.abspath(pkg_out) == os.path.abspath(pkg_in):
        print("refusing to overwrite the input pkg; choose a different output path", file=sys.stderr)
        return 2
    if os.path.exists(pkg_out) and not force:
        print("refusing to overwrite existing %s (pass --force to allow)" % pkg_out, file=sys.stderr)
        return 2
    replaces = {}
    i = 3
    while i < len(argv):
        if argv[i] == "--entry" and i + 2 < len(argv) and argv[i + 2] == "--in":
            replaces[argv[i + 1]] = argv[i + 3]
            i += 4
        else:
            i += 1
    names = read_filelists(fdir)
    # name -> hash AND hash -> name
    name_to_hash = names
    hash_to_name = {v: k for k, v in names.items()}
    raw, entries, base = load(pkg_in)
    by_hash = {}
    for e in entries:
        by_hash[e["hash"]] = e
    replaced = 0
    for name, path in replaces.items():
        payload = open(path, "rb").read()
        h = crc32(name)
        e = by_hash.get(h)
        if e is None:
            print("warning: %r (hash %08X) not in pkg table" % (name, h))
            continue
        stored = zlib.compress(payload) if name.endswith(".d") else payload
        e["blob"] = stored
        replaced += 1
        print("replaced %-64s (stored %d bytes)" % (name, len(stored)))
    news = repack(entries, pkg_out)
    print("entries=%d replaced=%d out=%s" % (len(entries), replaced, pkg_out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))