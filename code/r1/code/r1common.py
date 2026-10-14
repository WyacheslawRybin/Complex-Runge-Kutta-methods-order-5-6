"""Paths of the R1 revision and access to the original code.

Layout (relative to the project root):
    paper/            the revised paper (crk_cck_paper.tex, .bib) with its tables/ and figures/
    code/r1/code/     the scripts of the revision (this folder), code/r1/data/ their data
    code/base/        the pipeline of the original submission: code/, data/, tables/, figures/
The scripts of the original submission are imported read-only; their data are not modified.
"""
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
R1_WORK = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(R1_WORK, "..", ".."))
ORIG = os.path.join(ROOT, "code", "base")
ORIG_CODE = os.path.join(ORIG, "code")
ORIG_DATA = os.path.join(ORIG, "data")
ORIG_TAB = os.path.join(ORIG, "tables")
R1 = PAPER = os.path.join(ROOT, "paper")
DATA = os.path.join(R1_WORK, "data")
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figures")
TEX = os.path.join(PAPER, "crk_cck_paper.tex")
for d in (DATA, TAB, FIG):
    os.makedirs(d, exist_ok=True)
if ORIG_CODE not in sys.path:
    sys.path.insert(0, ORIG_CODE)
