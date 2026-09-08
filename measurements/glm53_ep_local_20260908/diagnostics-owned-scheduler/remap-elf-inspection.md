All 24 remap cubins differ only in the source-file modification timestamp in `.debug_line` and `.nv.merc.debug_line`.

Each cubin has four changed bytes: section-relative offsets 136 and 137 in each of those two sections. Both sections have SHF_ALLOC clear.

ELF headers, complete section metadata, program headers, executable and allocated sections, relocations, source paths, line programs, and every other byte are identical. All 24 PTX artifacts match their own receipts and each other.

The DWARF v2 file-table mtime decodes from 1788872668 to 1788880342. CPU17 matches the archived source stat (`mtime_ns=1788880342292681361`). CPU16's decoded timestamp has not been independently checked against a source stat by this inspector.

Source: `/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/glm53_ep_route_remap.py`, length 5493, SHA-256 `9d83120fb20e4aae382e3971a86c1a3b6970e591dd31ee5d7fc3e1a2ca521339`. The source hash is bound by both compilation receipts and CPU17 archived source bytes.

[DWARF v2 section 6.2 file table specification](https://dwarfstd.org/doc/dwarf-2.0.0.pdf) identifies the ULEB128 timestamp field after the filename and directory index.

Six local byte-mutation checks were rejected: executable text, filename, line program, file size, mtime inconsistent with the archived source stat, and ELF flags. No accelerator or repository module is imported.

This report classifies the binary difference; it does not replace the original exact-byte rejection, admit the collecting directory, or establish GPU numerics/performance acceptance.

Per-file original/stored hashes, exact differing bytes, all section hashes and decoded DWARF fields are in the adjacent JSON.
