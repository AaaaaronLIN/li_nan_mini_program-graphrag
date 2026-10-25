"""Render research_brief.md to a PDF, auto-fitting to a single page when possible."""
import re
import textwrap
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

src = Path("research_brief.md")
out = Path("research_brief.pdf")

WRAP = 108          # characters per line
BODY_FS = 8.8
STEP_BODY = 0.0142
STEP_BLANK = 0.009
STEP_H2_EXTRA = 0.010
STEP_TITLE = 0.032
TOP = 0.958
BOTTOM = 0.048


def parse(md_path: Path, wrap: int):
    lines = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        s = re.sub(r"\*\*(.*?)\*\*", r"\1", line.rstrip())
        s = re.sub(r"`(.*?)`", r"\1", s)
        if s.startswith("```"):
            continue
        if not s:
            lines.append(("BLANK", ""))
        elif s.startswith("# "):
            lines.append(("TITLE", s[2:].strip()))
        elif s.startswith("## "):
            lines.append(("H2", s[3:].strip()))
        elif s.startswith("- "):
            wrapped = textwrap.wrap(s[2:].strip(), width=wrap)
            lines.append(("BULLET", wrapped[0]))
            lines += [("INDENT", w) for w in wrapped[1:]]
        else:
            lines += [("BODY", w) for w in textwrap.wrap(s.strip(), width=wrap)]
    return lines


def height(lines):
    h = 0.0
    for kind, _ in lines:
        if kind == "TITLE":
            h += STEP_TITLE
        elif kind == "H2":
            h += STEP_H2_EXTRA + STEP_BODY
        elif kind == "BLANK":
            h += STEP_BLANK
        else:
            h += STEP_BODY
    return h


def render(lines, pdf):
    fig = Figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = TOP
    for kind, text in lines:
        if kind == "TITLE":
            fig.text(0.08, y, text, fontsize=15.5, weight="bold", color="#8b0000", va="top")
            y -= STEP_TITLE
        elif kind == "H2":
            y -= STEP_H2_EXTRA
            fig.text(0.08, y, text, fontsize=10.5, weight="bold", color="#8b0000", va="top")
            y -= STEP_BODY
        elif kind == "BULLET":
            fig.text(0.10, y, "\u2022 " + text, fontsize=BODY_FS, va="top")
            y -= STEP_BODY
        elif kind == "INDENT":
            fig.text(0.13, y, text, fontsize=BODY_FS, va="top")
            y -= STEP_BODY
        elif kind == "BODY":
            fig.text(0.08, y, text, fontsize=BODY_FS, va="top")
            y -= STEP_BODY
        else:
            y -= STEP_BLANK
    fig.text(0.08, 0.035, "Project: https://github.com/AaaaaronLIN/li_nan_mini_program-graphrag",
             fontsize=7.5, color="#666666")
    pdf.savefig(fig)
    return y


lines = parse(src, WRAP)
total = height(lines)
one_page = TOP - total >= BOTTOM
print(f"content height {total:.3f} | one page: {one_page}")

with PdfPages(out) as pdf:
    if one_page:
        render(lines, pdf)
    else:
        # fall back to balanced two pages, splitting at a section heading
        half = len(lines) // 2
        while half < len(lines) and lines[half][0] != "H2":
            half += 1
        render(lines[:half], pdf)
        render(lines[half:], pdf)

print("wrote", out)
