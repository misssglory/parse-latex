import os
import difflib
import shutil
import subprocess
import tempfile


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

    tex = (
        r"""
\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\begin{document}
$%s$
\end{document}
"""
        % formula
    )

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
