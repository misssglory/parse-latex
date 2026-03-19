import os
import re
import cv2
import difflib
import shutil
import subprocess
import tempfile
import numpy as np


def char_diff(gt: str, pred: str) -> str:
    out = []
    sm = difflib.SequenceMatcher(a=gt, b=pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(gt[i1:i2])
        elif tag == "delete":
            out.append(f"[-{gt[i1:i2]}-]")
        elif tag == "insert":
            out.append(f"[+{pred[j1:j2]}+]")
        elif tag == "replace":
            out.append(f"[-{gt[i1:i2]}-][+{pred[j1:j2]}+]")
    return "".join(out)


def compile_latex_formula(formula: str, workdir=None):
    if shutil.which("pdflatex") is None:
        return False, "pdflatex_not_found"

    own_dir = False
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="im2latex_")
        own_dir = True

    tex = r"""
\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\begin{document}
$%s$
\end{document}
""" % formula

    tex_path = os.path.join(workdir, "sample.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    try:
        p = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "sample.tex"],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            text=True,
        )
        ok = p.returncode == 0
        msg = p.stdout[-2000:]
    except Exception as e:
        ok = False
        msg = str(e)

    if own_dir:
        shutil.rmtree(workdir, ignore_errors=True)
    return ok, msg


def render_formula_to_image(formula: str, out_png: str):
    workdir = tempfile.mkdtemp(prefix="im2latex_render_")
    tex = r"""
\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\begin{document}
$%s$
\end{document}
""" % formula
    tex_path = os.path.join(workdir, "sample.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    try:
        p = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "sample.tex"],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            text=True,
        )
        if p.returncode != 0:
            return False
        if shutil.which("convert") is None and shutil.which("magick") is None:
            return False
        pdf_path = os.path.join(workdir, "sample.pdf")
        cmd = ["magick", "-density", "200", pdf_path, "-quality", "100", out_png] \
            if shutil.which("magick") else \
            ["convert", "-density", "200", pdf_path, "-quality", "100", out_png]
        p2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, text=True)
        return p2.returncode == 0 and os.path.exists(out_png)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def exact_image_match(gt_path: str, pred_render_path: str):
    if not (os.path.exists(gt_path) and os.path.exists(pred_render_path)):
        return False
    a = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(pred_render_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)