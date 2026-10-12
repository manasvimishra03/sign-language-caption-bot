"""Bridge between the Caption Bot (Python) and the RV32I processor built in Logisim Evolution.

Facts read from Risc-five.circ (circuit "rv32i_microprocessor"):
  * ROM and RAM are both 4096 words x 32 bit.
  * ROM address = PC[13:2]; RAM address = ALU_result[13:2]  -> word addressed.
  * Only lw / sw exist (no byte or half-word access) -> one character per 32-bit word.
  * All 32 registers are writable (x0 is NOT hard-wired to 0), so the program below
    never uses x0 as a destination register.

RAM layout used by the caption program
  word 0    OUT  number of words in the caption
  word 1    OUT  number of characters
  word 2    OUT  checksum = sum of all character codes
  word 3    -
  word 4..  IN   caption, one character code per word, terminated by a 0 word

Run `python rv32i_bridge.py` to self-test and (re)generate caption_rom.mem and
sample_ram_image.mem.
"""
from __future__ import annotations

import re
from pathlib import Path

RAM_WORDS = 4096
IN_BASE_WORD = 4
MAX_CHARS = 600  # keeps the Logisim demo short; hard limit is RAM_WORDS - IN_BASE_WORD - 1
M32 = 0xFFFFFFFF

# Only instructions already used by the existing ROM program (addi, add, lw, sw, beq, jal).
CAPTION_PROGRAM = """
        addi x3, x0, 16        # ptr = byte address of RAM word 4
        addi x4, x0, 0         # space counter
        addi x6, x0, 0         # character counter
        addi x8, x0, 32        # ASCII ' '
        addi x10, x0, 0        # checksum
loop:   lw   x5, 0(x3)         # next character
        beq  x5, x0, done      # 0 terminator -> finished
        add  x10, x10, x5      # checksum += char
        addi x6, x6, 1         # chars += 1
        beq  x5, x8, space     # char == ' ' ?
next:   addi x3, x3, 4         # ptr += 4 (next word)
        jal  x7, loop
space:  addi x4, x4, 1         # spaces += 1
        jal  x7, next
done:   beq  x6, x0, store     # empty caption -> words stays 0
        addi x4, x4, 1         # words = spaces + 1
store:  sw   x4, 0(x0)         # RAM[0] = words
        sw   x6, 4(x0)         # RAM[1] = chars
        sw   x10, 8(x0)        # RAM[2] = checksum
halt:   jal  x7, halt          # spin here
"""


# ---------------------------------------------------------------- caption <-> RAM image
def normalize_caption(text: str) -> str:
    return " ".join(text.split())


def build_ram_image(text: str, max_chars: int = MAX_CHARS) -> tuple[str, list[int]]:
    """Return (payload actually sent to the processor, RAM words)."""
    payload = normalize_caption(text)[:max_chars].rstrip()
    return payload, [0] * IN_BASE_WORD + [ord(c) for c in payload] + [0]


def expected_results(payload: str) -> dict:
    return {
        "words": len(payload.split()),
        "characters": len(payload),
        "checksum": sum(map(ord, payload)) & M32,
    }


def estimate_cycles(payload: str) -> int:
    """Instructions the program executes before it reaches the halt loop."""
    spaces = payload.count(" ")
    others = len(payload) - spaces
    return 11 + 7 * others + 9 * spaces + (1 if payload else 0)


def format_mem(words, per_line: int = 8) -> str:
    """Logisim 'v2.0 raw' memory image."""
    lines = ["v2.0 raw"]
    for i in range(0, len(words), per_line):
        lines.append(" ".join(f"{w & M32:x}" for w in words[i:i + per_line]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- tiny assembler
_MEM_OPERAND = re.compile(r"^(-?\w+)\((x\d+)\)$")


def _reg(tok: str) -> int:
    m = re.fullmatch(r"x(\d+)", tok.strip())
    if not m or int(m.group(1)) > 31:
        raise ValueError(f"bad register: {tok!r}")
    return int(m.group(1))


def _mem(tok: str) -> tuple[int, int]:
    m = _MEM_OPERAND.match(tok.strip())
    if not m:
        raise ValueError(f"bad memory operand: {tok!r}")
    return int(m.group(1), 0), _reg(m.group(2))


def _check12(imm: int) -> None:
    if not -2048 <= imm <= 2047:
        raise ValueError(f"immediate {imm} does not fit in 12 bits")


def _enc_i(op, f3, rd, rs1, imm):
    _check12(imm)
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def _enc_s(rs2, rs1, imm):
    _check12(imm)
    return (((imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | ((imm & 0x1F) << 7) | 0x23


def _enc_b(rs1, rs2, off):
    if off % 2 or not -4096 <= off <= 4094:
        raise ValueError(f"bad branch offset {off}")
    i = off & 0x1FFF
    return ((((i >> 12) & 1) << 31) | (((i >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15)
            | (((i >> 1) & 0xF) << 8) | (((i >> 11) & 1) << 7) | 0x63)


def _enc_j(rd, off):
    if off % 2 or not -(1 << 20) <= off < (1 << 20):
        raise ValueError(f"bad jump offset {off}")
    i = off & 0x1FFFFF
    return ((((i >> 20) & 1) << 31) | (((i >> 1) & 0x3FF) << 21) | (((i >> 11) & 1) << 20)
            | (((i >> 12) & 0xFF) << 12) | (rd << 7) | 0x6F)


def assemble(src: str) -> list[int]:
    items, labels, pc = [], {}, 0
    for raw in src.splitlines():
        line = raw.split("#", 1)[0].strip()
        while ":" in line:
            name, line = line.split(":", 1)
            labels[name.strip()] = pc
            line = line.strip()
        if not line:
            continue
        mnem, _, rest = line.partition(" ")
        args = [a.strip() for a in rest.split(",")] if rest.strip() else []
        items.append((pc, mnem.lower(), args))
        pc += 4

    def target(tok: str, here: int) -> int:
        tok = tok.strip()
        return labels[tok] - here if tok in labels else int(tok, 0)

    out = []
    for pc, m, a in items:
        if m == "addi":
            out.append(_enc_i(0x13, 0, _reg(a[0]), _reg(a[1]), int(a[2], 0)))
        elif m == "add":
            out.append((_reg(a[2]) << 20) | (_reg(a[1]) << 15) | (_reg(a[0]) << 7) | 0x33)
        elif m == "lw":
            off, base = _mem(a[1])
            out.append(_enc_i(0x03, 2, _reg(a[0]), base, off))
        elif m == "sw":
            off, base = _mem(a[1])
            out.append(_enc_s(_reg(a[0]), base, off))
        elif m == "beq":
            out.append(_enc_b(_reg(a[0]), _reg(a[1]), target(a[2], pc)))
        elif m == "jal":
            out.append(_enc_j(_reg(a[0]), target(a[1], pc)))
        else:
            raise ValueError(f"unsupported instruction: {m}")
    return out


def rom_image() -> str:
    return format_mem(assemble(CAPTION_PROGRAM))


# ---------------------------------------------------------------- emulator (dev-time check only)
def execute(rom, ram_init, max_steps: int = 500_000):
    """Minimal model of the Logisim processor's word addressing. Used ONLY to check the
    program offline; the demo result must come from Logisim itself."""
    rom = list(rom) + [0] * (RAM_WORDS - len(rom))
    ram = list(ram_init) + [0] * (RAM_WORDS - len(ram_init))
    x = [0] * 32
    pc, steps = 0, 0

    def sx(v, bits):
        return v - (1 << bits) if v & (1 << (bits - 1)) else v

    def write(rd, val):
        if rd == 0:
            raise AssertionError("instruction writes x0 (not hard-wired in this register file)")
        x[rd] = val & M32

    while steps < max_steps:
        ins = rom[(pc >> 2) & 0xFFF]
        op, rd, f3 = ins & 0x7F, (ins >> 7) & 31, (ins >> 12) & 7
        rs1, rs2 = (ins >> 15) & 31, (ins >> 20) & 31
        npc = (pc + 4) & M32
        if op == 0x13 and f3 == 0:
            write(rd, x[rs1] + sx(ins >> 20, 12))
        elif op == 0x33 and f3 == 0 and (ins >> 25) == 0:
            write(rd, x[rs1] + x[rs2])
        elif op == 0x03 and f3 == 2:
            write(rd, ram[((x[rs1] + sx(ins >> 20, 12)) & M32) >> 2 & 0xFFF])
        elif op == 0x23 and f3 == 2:
            imm = sx(((ins >> 25) << 5) | ((ins >> 7) & 0x1F), 12)
            ram[((x[rs1] + imm) & M32) >> 2 & 0xFFF] = x[rs2]
        elif op == 0x63 and f3 == 0:
            imm = sx((((ins >> 31) & 1) << 12) | (((ins >> 7) & 1) << 11)
                     | (((ins >> 25) & 0x3F) << 5) | (((ins >> 8) & 0xF) << 1), 13)
            if x[rs1] == x[rs2]:
                npc = (pc + imm) & M32
        elif op == 0x6F:
            imm = sx((((ins >> 31) & 1) << 20) | (((ins >> 12) & 0xFF) << 12)
                     | (((ins >> 20) & 1) << 11) | (((ins >> 21) & 0x3FF) << 1), 21)
            if imm == 0:
                return ram, steps
            write(rd, pc + 4)
            npc = (pc + imm) & M32
        else:
            raise NotImplementedError(f"unsupported instruction {ins:#010x} at pc={pc:#x}")
        steps += 1
        pc = npc
    raise RuntimeError("program did not halt")


# ---------------------------------------------------------------- self-test
def _selftest() -> None:
    # 1. Assembler must reproduce the words already sitting in Risc-five.circ's ROM.
    known = [
        ("addi x2, x0, 1", 0x00100113), ("addi x10, x0, 4", 0x00400513),
        ("addi x6, x0, 40", 0x02800313), ("addi x3, x0, 0", 0x00000193),
        ("addi x4, x3, 4", 0x00418213), ("sw x1, 256(x3)", 0x1011A023),
        ("sw x2, 256(x4)", 0x10222023), ("addi x14, x0, 8", 0x00800713),
        ("addi x5, x0, 8", 0x00800293), ("addi x13, x0, 8", 0x00800693),
        ("addi x15, x0, 4", 0x00400793), ("addi x9, x0, 4", 0x00400493),
        ("add x8, x1, x2", 0x00208433), ("beq x5, x6, 32", 0x02628063),
        ("add x12, x0, x8", 0x00800633), ("sw x12, 256(x5)", 0x10C2A023),
        ("lw x11, 256(x9)", 0x1004A583), ("add x8, x11, x8", 0x00858433),
        ("addi x5, x5, 4", 0x00428293), ("addi x9, x9, 4", 0x00448493),
        ("jal x7, -28", 0xFE5FF3EF), ("beq x3, x6, 16", 0x00618863),
        ("lw x16, 256(x3)", 0x1001A803), ("addi x3, x3, 4", 0x00418193),
        ("jal x7, -12", 0xFF5FF3EF),
    ]
    for src, want in known:
        got = assemble(src)[0]
        assert got == want, f"{src}: got {got:#010x}, want {want:#010x}"
    print(f"assembler reproduces {len(known)}/{len(known)} instruction words from your existing ROM")

    # 2. The caption program, run on a model of your processor's addressing.
    rom = assemble(CAPTION_PROGRAM)
    cases = [
        "", "hello", "Meet me at the college gate at 10 AM.", "a b c d e f g h i j",
        "नमस्ते दोस्तों, आप कैसे हैं?", " ".join(["lorem"] * 100), "word " * 400,
        "   spaced    out \n text  ",
    ]
    for text in cases:
        payload, ram_words = build_ram_image(text)
        ram, steps = execute(rom, ram_words)
        got = {"words": ram[0], "characters": ram[1], "checksum": ram[2]}
        assert got == expected_results(payload), (text[:30], got, expected_results(payload))
        assert steps == estimate_cycles(payload), (text[:30], steps, estimate_cycles(payload))
    print(f"program matches Python on {len(cases)} captions (incl. empty, Hindi, truncated); cycle estimate exact")

    # 3. Files for Logisim.
    here = Path(__file__).resolve().parent
    (here / "caption_rom.mem").write_text(rom_image())
    sample = "Meet me at the college gate at 10 AM."
    payload, words = build_ram_image(sample)
    (here / "sample_ram_image.mem").write_text(format_mem(words))
    exp = expected_results(payload)
    print("\nROM image (caption_rom.mem):")
    print(rom_image())
    print(f"sample caption: {sample!r}")
    print(f"expected RAM word0/1/2 = {exp['words']:#x} / {exp['characters']:#x} / {exp['checksum']:#x}"
          f"  (= {exp['words']} / {exp['characters']} / {exp['checksum']}), cycles = {estimate_cycles(payload)}")


if __name__ == "__main__":
    _selftest()
