#!/usr/bin/env python3
"""Replace non-ASCII characters in notebook_final.tex that cause xelatex blank glyphs."""
import re, sys

with open("notebook_final.tex", encoding="utf-8") as f:
    src = f.read()

# Characters that lmmono can't render → safe ASCII/LaTeX substitutes
replacements = {
    "\u2500": "-",   # ─  box horizontal
    "\u2502": "|",   # │  box vertical
    "\u250c": "+",   # ┌
    "\u2510": "+",   # ┐
    "\u2514": "+",   # └
    "\u2518": "+",   # ┘
    "\u251c": "+",   # ├
    "\u2524": "+",   # ┤
    "\u252c": "+",   # ┬
    "\u2534": "+",   # ┴
    "\u253c": "+",   # ┼
    "\u2550": "=",   # ═
    "\u2551": "||",  # ║
    "\u2026": "...", # …
    "\u2019": "'",   # '
    "\u2018": "'",   # '
    "\u201c": '"',   # "
    "\u201d": '"',   # "
    "\u2013": "--",  # –
    "\u2014": "---", # —
    "\u00d7": r"\times ",  # ×
    "\u2248": r"\approx ", # ≈
    "\u2260": r"\neq ",    # ≠
    "\u2264": r"\leq ",    # ≤
    "\u2265": r"\geq ",    # ≥
    "\u03b1": r"\alpha ",  # α
    "\u03b2": r"\beta ",   # β
    "\u03b3": r"\gamma ",  # γ
    "\u03b4": r"\delta ",  # δ
    "\u03b5": r"\epsilon ",# ε
    "\u03b7": r"\eta ",    # η
    "\u03bb": r"\lambda ", # λ
    "\u03bc": r"\mu ",     # μ
    "\u03c3": r"\sigma ",  # σ
    "\u03c6": r"\phi ",    # φ
    "\u03c8": r"\psi ",    # ψ
    "\u03c9": r"\omega ",  # ω
    "\u0394": r"\Delta ",  # Δ
    "\u03a3": r"\Sigma ",  # Σ
    "\u2016": r"\|",       # ‖ double vertical (norm)
    "\u2081": "1",         # subscript 1
    "\u2082": "2",         # subscript 2
    "\u2083": "3",         # subscript 3
    "\u00b2": "^2",        # superscript 2
    "\u00b3": "^3",        # superscript 3
    "\u2192": r"\rightarrow ", # →
    "\u2190": r"\leftarrow ",  # ←
    "\u00e9": "e",         # é
    "\u00e0": "a",         # à
    "\u00fc": "u",         # ü
    "\u00e4": "a",         # ä
    "\u00f6": "o",         # ö
    "\u211d": "R",         # ℝ real numbers
    "\u2208": r"\in ",     # ∈
    "\u2212": "-",         # − minus sign
    "\u221e": r"\infty ",  # ∞
    "\u00b1": r"\pm ",     # ±
    "\u0302": "",          # combining circumflex (drop)
    "\u00c3": "A",         # Ã (encoding artifact from Ã²X)
    "\u00b2": "^2",        # ²
    "\U0001d6e5": "Delta", # 𝛥 mathematical bold italic capital delta
    "\U0001d6e6": "Epsilon",
    "\U0001d6e7": "Zeta",
    "\U0001d6e8": "Eta",
    "\U0001d6e9": "Theta",
    "\U0001d6f3": "delta", # 𝛿
    "\U0001d6f4": "epsilon",
    "\U0001d6fc": "alpha",
    "\U0001d6fd": "beta",
    "\U0001d6fe": "gamma",
    "\U0001d70e": "sigma",
    "\U0001d707": "eta",
    "\U0001d707": "eta",
    "\uff08": "(",          # （ fullwidth left parenthesis
    "\uff09": ")",          # ） fullwidth right parenthesis
    "\uff0c": ",",          # ，fullwidth comma
    "\u00a7": "S",          # § section sign
}

for char, repl in replacements.items():
    src = src.replace(char, repl)

with open("notebook_final.tex", "w", encoding="utf-8") as f:
    f.write(src)

# ── Pass 2: inside Verbatim blocks, undo LaTeX commands → plain ASCII ────────
# (Verbatim blocks with commandchars process \Delta etc. as LaTeX → bad glyphs)
VERBATIM_RE = re.compile(
    r"(\\begin\{Verbatim\}.*?\\end\{Verbatim\})", re.DOTALL
)
verbatim_replacements = {
    r"\Delta ":  "Delta",
    r"\Sigma ":  "Sigma",
    r"\alpha ":  "alpha",
    r"\beta ":   "beta",
    r"\gamma ":  "gamma",
    r"\delta ":  "delta",
    r"\epsilon ":"epsilon",
    r"\eta ":    "eta",
    r"\lambda ": "lambda",
    r"\mu ":     "mu",
    r"\sigma ":  "sigma",
    r"\phi ":    "phi",
    r"\psi ":    "psi",
    r"\omega ":  "omega",
    r"\times ":  "x",
    r"\approx ": "~",
    r"\leq ":    "<=",
    r"\geq ":    ">=",
    r"\neq ":    "!=",
    r"\in ":     "in",
    r"\infty ":  "inf",
    r"\pm ":     "+/-",
    r"\rightarrow ": "->",
    r"\leftarrow ":  "<-",
    r"\|":       "||",
}

def fix_verbatim(m):
    block = m.group(1)
    for latex_cmd, ascii_rep in verbatim_replacements.items():
        block = block.replace(latex_cmd, ascii_rep)
    return block

src = VERBATIM_RE.sub(fix_verbatim, src)

with open("notebook_final.tex", "w", encoding="utf-8") as f:
    f.write(src)
print("Done.")

# Report any remaining non-ASCII in Verbatim blocks
non_ascii = {}
for v in VERBATIM_RE.findall(src):
    for c in v:
        if ord(c) > 127:
            non_ascii[c] = hex(ord(c))
if non_ascii:
    print(f"Remaining non-ASCII in verbatim ({len(non_ascii)} chars):")
    for c, h in sorted(non_ascii.items(), key=lambda x: x[1])[:30]:
        print(f"  {h}  {c}")
else:
    print("No remaining non-ASCII in verbatim blocks.")
