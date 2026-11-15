"""Proofreading helper: the main text of paper/main.tex with every generated number macro replaced by its
value in [[...]], and a list of macros used in the text but not defined in numbers.tex.
Usage: python expand_paper.py [--all]   (--all: whole document incl. appendix)"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (PAPER / "numbers.tex").read_text(encoding="utf-8")))
s = (PAPER / "main.tex").read_text(encoding="utf-8")
if "--all" not in sys.argv:
    s = s.split("\\begin{abstract}", 1)[1].split("\\label{lastmainpage}", 1)[0]
used = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", s))
latex_builtin = {"Delta"}
print("UNDEFINED MACROS:", sorted(k for k in used if k not in mac and k not in latex_builtin))
print(re.sub(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", lambda m: "[[" + mac[m.group(1)] + "]]" if m.group(1) in mac else m.group(0), s))
