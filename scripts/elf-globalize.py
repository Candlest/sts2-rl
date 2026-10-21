#!/usr/bin/env python3
"""Promote named symbols in an ELF object file to GLOBAL binding + DEFAULT
visibility by patching .symtab entries in place.

GNU objcopy --globalize-symbol handles the binding but cannot change symbol
visibility; ILC 9.x emits [UnmanagedCallersOnly] entry points as
GLOBAL/HIDDEN (hidden symbols never reach .dynsym), so both must be fixed.

Usage: elf-globalize.py <elf-object> <symbol-name> [<symbol-name> ...]
"""

from __future__ import annotations

import struct
import sys


def fail(msg: str) -> None:
    print(f"elf-globalize: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 3:
        fail("usage: elf-globalize.py <elf-object> <sym> [...]")
    path, targets = sys.argv[1], set(sys.argv[2:])
    patched: set[str] = set()

    with open(path, "rb+") as f:
        ident = f.read(16)
        if ident[:4] != b"\x7fELF":
            fail("not an ELF file")
        if ident[4] != 2:  # EI_CLASS
            fail("not a 64-bit ELF")
        if ident[5] != 1:  # EI_DATA = LSB
            fail("not a little-endian ELF")

        f.seek(0)
        ehdr = struct.unpack("<16sHHIQQQIHHHHHH", f.read(64))
        e_shoff = ehdr[6]
        e_shentsize = ehdr[11]
        e_shnum = ehdr[12]
        e_shstrndx = ehdr[13]

        sections = []
        for i in range(e_shnum):
            f.seek(e_shoff + i * e_shentsize)
            sh = struct.unpack("<IIQQQQIIQQ", f.read(64))
            sections.append(
                {
                    "name_off": sh[0],
                    "type": sh[1],
                    "offset": sh[4],
                    "size": sh[5],
                    "link": sh[6],
                    "entsize": sh[9],
                }
            )

        shstr = sections[e_shstrndx]

        def section_name(sec: dict) -> str:
            f.seek(shstr["offset"] + sec["name_off"])
            out = bytearray()
            while True:
                b = f.read(1)
                if b == b"\x00":
                    return out.decode()
                out += b

        symtab = next((s for s in sections if s["type"] == 2), None)  # SHT_SYMTAB
        if symtab is None:
            fail("no .symtab section")
        strtab = sections[symtab["link"]]
        if section_name(strtab) != ".strtab":
            fail("symtab link target is not .strtab")

        count = symtab["size"] // 24  # Elf64_Sym
        for idx in range(count):
            entry_off = symtab["offset"] + idx * 24
            f.seek(entry_off)
            st_name, st_info, st_other = struct.unpack("<IBB", f.read(6))
            f.seek(strtab["offset"] + st_name)
            name = bytearray()
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            name = name.decode()
            if name not in targets:
                continue
            new_info = (st_info & 0xF0) | 0x10  # STB_GLOBAL
            new_other = st_other & ~0x03  # STV_DEFAULT
            f.seek(entry_off + 4)
            f.write(struct.pack("<BB", new_info, new_other))
            patched.add(name)

    missing = targets - patched
    if missing:
        fail(f"symbols not found: {sorted(missing)[:5]}")
    print(f"elf-globalize: patched {len(patched)} symbols in {path}")


if __name__ == "__main__":
    main()
