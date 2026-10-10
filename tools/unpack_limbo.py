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
    for idx, (name_hash, entry_off, size) in enumerate(entries):
        blob = raw[base + entry_off : base + entry_off + size]
        name = names.get(name_hash)
        if name is None:
            name = os.path.join("__UNKNOWN", "%08X" % name_hash)
        else:
            named += 1
            if name.endswith(".d"):
                inflated = inflate(blob)
                if inflated is not None:
                    blob = inflated
                    name = name[:-2]
        target = os.path.join(outdir, name.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as out:
            out.write(blob)
    return count, named


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
    total, named = unpack(pkg_path, names, outdir)
    print("entries=%d named=%d (%.0f%%)" % (total, named, 100.0 * named / total))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))