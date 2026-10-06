# -*- coding: utf-8 -*-
"""Journal artwork for the PHL manuscript (matplotlib -> vector PDF + 600 dpi TIFF).

Run:  python3.8 make_figs.py [1|2|all]        (python3.8 carries matplotlib 3.5 on this host)

Fig1_PHL_architecture   : schematic method overview, 178 x 104 mm (double column)
Fig2_degree_diagnostics : degree-stratified loss diagnostics, 178 x 90 mm

OUP Bioinformatics artwork rules honoured here
  * drawn at final print size: 178 mm double-column width
  * Arial/Helvetica-class lettering (Liberation Sans, metric-identical to Arial), 6.5-7 pt body,
    7.5 pt bold panel headings, 10 pt bold panel letters; line weights 0.3-1 pt; TrueType fonts embedded
  * schematic only: no figure title, caption, footnote or result numbers inside the artwork
  * RGB, white background, nothing identified by colour alone (shapes / text labels carry identity)
  * deliverables: vector PDF (preferred for line art) and 600 dpi LZW TIFF (combination art)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, RegularPolygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Liberation Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold", "mathtext.default": "regular",
    "axes.linewidth": 0.5, "lines.linewidth": 0.75,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "savefig.facecolor": "white",
})

INK, INK2, MUTED, HAIR = "#000000", "#404040", "#707070", "#B0B0B0"
BLUE, ORANGE, GREEN, PURPLE = "#2A78D6", "#E8652E", "#159C6B", "#8E5BC0"
BLUE_T, ORANGE_T, GREEN_T, PURPLE_T, GREY_T = "#DCEAFB", "#FCE4D8", "#DAF1E7", "#EBE1F6", "#F0F0F0"
P_LETTER, P_HEAD, P_BODY, P_SMALL = 10, 7.5, 7, 6.5
MM = 25.4


# ----------------------------------------------------------------------------- drawing helpers
def canvas(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm / MM, h_mm / MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)  # y grows downwards, like a page
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, size=P_BODY, color=INK, ha="left", va="center", bold=False, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight="bold" if bold else "normal", linespacing=1.15, **kw)


def rbox(ax, x, y, w, h, fc="white", ec=INK2, lw=0.5, r=1.0, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=0.75, ms=5, ls="-", z=3, style="-|>", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        zorder=z, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def line(ax, x1, y1, x2, y2, color=INK2, lw=0.75, ls="-", z=2, rad=0.0):
    return arrow(ax, x1, y1, x2, y2, color=color, lw=lw, ls=ls, z=z, style="-", rad=rad)


def blob(ax, pts, fc, pad_mm=3.4, z=0):
    """Hyperedge drawn as a rounded region enclosing its member nodes."""
    pts = np.asarray(pts, float)
    lw_pt = 2 * pad_mm * 72 / MM
    if len(pts) == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=fc, lw=lw_pt, solid_capstyle="round", zorder=z)
    else:
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=fc, lw=lw_pt, joinstyle="round", zorder=z))


def node_m(ax, x, y, r=1.5, fc=BLUE, z=5):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="white", lw=0.5, zorder=z))


def node_d(ax, x, y, r=1.5, ec=BLUE, z=5):
    ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, fc="white", ec=ec, lw=1.0, zorder=z))


def node_hex(ax, x, y, r=1.9, fc=GREEN, z=5):
    ax.add_patch(RegularPolygon((x, y), 6, radius=r, fc=fc, ec="white", lw=0.5, zorder=z))


def node_dia(ax, x, y, r=1.9, fc=GREEN, z=5):
    ax.add_patch(RegularPolygon((x, y), 4, radius=r, fc=fc, ec="white", lw=0.5, zorder=z))


def node_bar(ax, x, y, fc=GREEN, z=5):
    """cross-phenotype degree glyph: three bars in a tile."""
    rbox(ax, x - 2.2, y - 1.8, 4.4, 3.6, fc=GREEN_T, ec=fc, lw=0.5, r=0.6, z=z)
    for k, h in enumerate((1.0, 2.2, 1.6)):
        ax.add_patch(Rectangle((x - 1.6 + k * 1.15, y + 1.2 - h), 0.8, h, fc=fc, ec="none", zorder=z + 1))


def panel_letter(ax, x, y, letter, heading=None):
    txt(ax, x, y, letter, size=P_LETTER, bold=True, va="baseline")
    if heading:
        txt(ax, x + 4.4, y, heading, size=P_HEAD, bold=True, va="baseline")


def grid(ax, x, y, ncol, nrow, cell, filled, fc, ec=HAIR, query=None, z=2):
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(Rectangle((x + j * cell, y + i * cell), cell, cell,
                                   fc=fc if (i, j) in filled else "white", ec=ec, lw=0.3, zorder=z))
    if query is not None:
        i, j = query
        ax.add_patch(Rectangle((x + j * cell, y + i * cell), cell, cell, fc="white", ec=INK, lw=0.6,
                               ls=(0, (1.5, 1)), zorder=z + 1))
        txt(ax, x + (j + 0.5) * cell, y + (i + 0.5) * cell, "?", size=P_SMALL, ha="center", bold=True)


# ============================================================================= FIGURE 1
def fig1(out):
    W, H = 178, 104
    fig, ax = canvas(W, H)
    XA, WA = 3, 35     # A  3-38
    XB, WB = 43, 59    # B 43-102
    XC, WC = 107, 37   # C 107-144
    XD, WD = 148, 28   # D 148-176
    HEAD_Y = 4.8

    panel_letter(ax, XA, HEAD_Y, "A", "Inputs")
    panel_letter(ax, XB, HEAD_Y, "B", "Relational hypergraph (support only)")
    panel_letter(ax, XC, HEAD_Y, "C", "Query-pair heuristics")
    panel_letter(ax, XD, HEAD_Y, "D", "Linear readout")

    # ----------------------------------------------------------------- A: inputs
    # A1 association matrix
    y1 = 9
    txt(ax, XA, y1, "Training positives (MDR or MDS)", bold=True, va="top")
    gx, gy, cell = XA + 6.5, y1 + 6.5, 2.3
    pos = {(0, 0), (0, 5), (1, 0), (1, 2), (2, 3), (2, 6), (3, 1), (3, 2), (4, 4), (4, 7), (5, 0), (5, 5), (6, 2),
           (6, 6), (1, 7), (5, 3), (3, 6)}
    ncol_, nrow_ = 8, 7
    grid(ax, gx, gy, ncol_, nrow_, cell, pos, BLUE, query=(0, 3))
    txt(ax, gx + ncol_ * cell / 2, gy - 1.0, "drugs", size=P_SMALL, color=MUTED, ha="center", va="bottom")
    txt(ax, gx - 1.0, gy + nrow_ * cell / 2, "miRNAs", size=P_SMALL, color=MUTED, ha="right", rotation=90)
    ly = gy + nrow_ * cell + 2.2
    ax.add_patch(Rectangle((XA + 0.5, ly - 1.1), 2.2, 2.2, fc=BLUE, ec="none", zorder=2))
    txt(ax, XA + 3.6, ly, "known association (training fold)", size=P_SMALL, color=INK2)
    ax.add_patch(Rectangle((XA + 0.5, ly + 2.4), 2.2, 2.2, fc="white", ec=INK, lw=0.5, ls=(0, (1.5, 1)), zorder=2))
    txt(ax, XA + 1.6, ly + 3.5, "?", size=P_SMALL, bold=True, ha="center")
    txt(ax, XA + 3.6, ly + 3.5, "query pair (held out)", size=P_SMALL, color=INK2)

    # A2 curation provenance
    y2 = 40
    txt(ax, XA, y2, "Curation provenance (iNcRD-HG)", bold=True, va="top")
    tx, ty = XA + 0.5, y2 + 5
    cols = [("pair", 8), ("PMID", 7), ("context terms", 19.5)]
    rows = [("$m_1$, $d_1$", "$P_1$", "MCF-7 · resistance"),
            ("$m_2$, $d_2$", "$P_1$", "MCF-7 · resistance"),
            ("$m_3$, $d_3$", "$P_2$", "A549 · sensitivity")]
    rh, tw = 3.5, sum(w for _, w in cols)
    ax.add_patch(Rectangle((tx, ty), tw, rh, fc=ORANGE_T, ec="none", zorder=1))
    x = tx
    for name, w in cols:
        txt(ax, x + 0.8, ty + rh / 2, name, size=P_SMALL, bold=True, color=INK2)
        x += w
    for r_i, row in enumerate(rows):
        yy = ty + rh * (r_i + 1)
        x = tx
        for (name, w), v in zip(cols, row):
            txt(ax, x + 0.8, yy + rh / 2, v, size=P_SMALL)
            x += w
        line(ax, tx, yy, tx + tw, yy, color=HAIR, lw=0.3)
    line(ax, tx, ty + rh * 4, tx + tw, ty + rh * 4, color=HAIR, lw=0.3)
    txt(ax, tx, ty + rh * 4 + 1.3, "per report: study identifier and its\ncell line / phenotype / disease terms",
        size=P_SMALL, color=INK2, va="top")

    # A3 label-free miRNA knowledge
    y3 = 71
    txt(ax, XA, y3, "Label-free miRNA knowledge", bold=True, va="top")
    hx, hy, hc = XA + 1.0, y3 + 5.5, 1.8
    hpos = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 3), (3, 2), (4, 1), (4, 3), (1, 4), (3, 4)}
    grid(ax, hx, hy, 5, 5, hc, hpos, GREEN)
    txt(ax, hx + 5 * hc + 2.5, hy + 2.5 * hc, "HMDD v4 miRNA–disease\nprofile", size=P_SMALL, color=INK2)
    node_hex(ax, XA + 5.5, y3 + 19.5)
    txt(ax, XA + 9.5, y3 + 19.5, "miRNA name family,\nseed family", size=P_SMALL, color=INK2)
    node_bar(ax, XA + 5.5, y3 + 26)
    txt(ax, XA + 9.5, y3 + 26.5, "degree in the other\nphenotype matrix", size=P_SMALL, color=INK2)
    for (ya, yb, c) in [(y1, y2 - 4, BLUE), (y2, y3 - 4, ORANGE), (y3, y3 + 29, GREEN)]:
        ax.add_patch(Rectangle((XA - 1.8, ya), 0.8, yb - ya, fc=c, ec="none", zorder=1))
    for yy in (y1 + 13, y2 + 11, y3 + 13):
        arrow(ax, XA + WA + 0.5, yy, XB - 1, yy)

    # ----------------------------------------------------------------- B: hypergraph
    sx, mx, dx = XB + 11, XB + 26, XB + 49
    m = {1: (mx, 18), 2: (mx, 33), 3: (mx, 48), 4: (mx, 63)}
    d = {1: (dx, 25.5), 2: (dx, 44), 3: (dx, 63)}
    F, Z, X = (sx, 25.5), (sx, 42), (sx, 63)
    blob(ax, [m[1], d[1], d[2], m[2]], ORANGE_T)
    blob(ax, [m[3], d[3], m[4]], PURPLE_T)
    txt(ax, (mx + dx) / 2, 9.2, "study hyperedge $P_1$:", size=P_SMALL, color=ORANGE, ha="center")
    txt(ax, (mx + dx) / 2, 12.0, "all pairs reported by one PMID", size=P_SMALL, color=ORANGE, ha="center")
    txt(ax, (mx + dx) / 2 + 3, 70.5, "context hyperedge:\nentities sharing a cell line,\nphenotype or disease term",
        size=P_SMALL, color=PURPLE, ha="center", va="top")
    for a, b in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)]:
        line(ax, *m[a], *d[b], color=INK, lw=0.8, z=3)
    line(ax, *m[1], *m[2], color=BLUE, lw=0.7, ls=(0, (0.8, 1.2)), rad=0.5, z=3)
    line(ax, *d[2], *d[3], color=BLUE, lw=0.7, ls=(0, (0.8, 1.2)), rad=-0.5, z=3)
    line(ax, *m[1], *d[2], color=INK2, lw=0.8, ls=(0, (2, 1.2)), z=3)
    qx, qy = m[1][0] + 0.68 * (d[2][0] - m[1][0]) + 2.2, m[1][1] + 0.68 * (d[2][1] - m[1][1]) - 2.2
    ax.add_patch(Circle((qx, qy), 1.7, fc="white", ec=INK2, lw=0.5, zorder=4))
    txt(ax, qx, qy, "?", size=P_SMALL, bold=True, ha="center", zorder=5)
    for p in (m[1], m[2]):
        line(ax, *F, *p, color=GREEN, lw=0.7, z=3)
    for p in (m[2], m[3]):
        line(ax, *Z, *p, color=GREEN, lw=0.7, z=3)
    line(ax, *X, *m[4], color=GREEN, lw=0.7, z=3)
    node_hex(ax, *F)
    node_dia(ax, *Z)
    node_bar(ax, *X)
    txt(ax, F[0] - 2.9, F[1], "family", size=P_SMALL, color=GREEN, ha="right")
    txt(ax, Z[0] - 2.9, Z[1], "HMDD\ndisease", size=P_SMALL, color=GREEN, ha="right")
    txt(ax, X[0], X[1] + 2.6, "other-matrix\ndegree", size=P_SMALL, color=GREEN, ha="center", va="top")
    for k, p in m.items():
        node_m(ax, *p)
        txt(ax, p[0] - 2.5, p[1], f"$m_{k}$", size=P_SMALL, ha="right", zorder=6)
    for k, p in d.items():
        node_d(ax, *p)
        txt(ax, p[0] + 2.5, p[1], f"$d_{k}$", size=P_SMALL, ha="left", zorder=6)
    # key for the edge / hyperedge styles (node identities are labelled in place)
    lx, ly, lh = XB + 1, 87, 12.5
    rbox(ax, lx, ly, WB - 2, lh, fc="white", ec=HAIR, lw=0.4, r=0.8, z=1)
    left = [("sup", "support edge"), ("knn", "kNN content edge"),
            ("qry", "query pair (held out)")]
    right = [("stu", "study hyperedge (PMID)"), ("ctx", "context hyperedge")]
    for k, (kind, lab) in enumerate(left):
        yy = ly + 2.4 + k * 3.7
        xx = lx + 4
        style = {"sup": (INK, 0.8, "-"), "knn": (BLUE, 0.7, (0, (0.8, 1.2))), "qry": (INK2, 0.8, (0, (2, 1.2)))}[kind]
        line(ax, xx - 2.5, yy, xx + 2.5, yy, color=style[0], lw=style[1], ls=style[2])
        txt(ax, xx + 3.8, yy, lab, size=P_SMALL, color=INK2)
    for k, (kind, lab) in enumerate(right):
        yy = ly + 2.4 + k * 3.7
        xx = lx + 31
        ax.add_patch(Rectangle((xx - 2.5, yy - 1.3), 5, 2.6, fc=ORANGE_T if kind == "stu" else PURPLE_T, ec="none",
                               zorder=2))
        txt(ax, xx + 3.8, yy, lab, size=P_SMALL, color=INK2)
    arrow(ax, XB + WB + 0.5, 44, XC - 1, 44)

    # ----------------------------------------------------------------- C: heuristic motifs
    groups = [
        (BLUE, "Graph context and diffusion (85)", "",
         "degree, common neighbours,\n3–7-step paths, PPR, Katz,\nrank-16 SVD, kNN support", "path"),
        (ORANGE, "Provenance (24)", "",
         "PMID bridging |P($m$) ∩ P($d$)|,\nco-study transfer, context-term\noverlap, record statistics", "prov"),
        (GREEN, "miRNA priors (19)", "",
         "family and seed mates of $m$\nlinked to $d$, HMDD disease\nbridge, cross-phenotype degree", "prior"),
    ]
    ys = [9, 37, 65]
    amp = 3.0
    for (col, name, n, desc, kind), y in zip(groups, ys):
        ax.add_patch(Rectangle((XC - 1.8, y), 0.8, 25.5, fc=col, ec="none", zorder=1))
        txt(ax, XC, y, name, bold=True, va="top")
        ym = y + 9.5
        a, b, c, e = (XC + 4, ym), (XC + 13, ym - amp), (XC + 22, ym + amp), (XC + 31, ym)
        line(ax, *a, *e, color=INK2, lw=0.8, ls=(0, (2, 1.2)), z=3)
        txt(ax, a[0] - 2.5, a[1], "$m$", size=P_SMALL, ha="right")
        txt(ax, e[0] + 2.5, e[1], "$d$", size=P_SMALL, ha="left")
        if kind == "path":
            for p, q in [(a, b), (b, c), (c, e)]:
                line(ax, *p, *q, color=INK, lw=0.8, z=3)
            node_m(ax, *a); node_d(ax, *b); node_m(ax, *c); node_d(ax, *e)
            txt(ax, b[0] + 2.4, b[1], "$d'$", size=P_SMALL)
            txt(ax, c[0] + 2.4, c[1], "$m'$", size=P_SMALL)
        elif kind == "prov":
            blob(ax, [b, c], ORANGE_T, pad_mm=2.8)
            line(ax, *a, *b, color=INK, lw=0.8, z=3)
            line(ax, *c, *e, color=INK, lw=0.8, z=3)
            node_m(ax, *a); node_d(ax, *b); node_m(ax, *c); node_d(ax, *e)
            txt(ax, b[0] + 2.4, b[1], "$d'$", size=P_SMALL)
            txt(ax, c[0] + 2.4, c[1], "$m'$", size=P_SMALL)
            txt(ax, XC + 26.5, ym - amp - 1.0, "same PMID", size=P_SMALL, color=ORANGE, ha="center", va="bottom")
        else:
            line(ax, *a, *b, color=GREEN, lw=0.7, z=3)
            line(ax, *b, *c, color=GREEN, lw=0.7, z=3)
            line(ax, *c, *e, color=INK, lw=0.8, z=3)
            node_m(ax, *a); node_hex(ax, *b); node_m(ax, *c); node_d(ax, *e)
            txt(ax, b[0] + 2.6, b[1], "family / HMDD disease", size=P_SMALL, color=GREEN)
            txt(ax, c[0] + 2.4, c[1], "$m'$", size=P_SMALL)
        txt(ax, XC, y + 15.8, desc, size=P_SMALL, color=INK2, va="top")

    # h(m,d) strip
    sy, sh = 94.5, 4.5
    x = XC
    for wv, col in zip([85, 24, 19], [BLUE, ORANGE, GREEN]):
        w = WC * wv / 128
        ax.add_patch(Rectangle((x, sy), w, sh, fc=col, ec="white", lw=0.5, zorder=2))
        txt(ax, x + w / 2, sy + sh / 2, str(wv), size=P_SMALL, color="white", ha="center", bold=True, zorder=3)
        x += w
    txt(ax, XC, sy - 1.2, "$\\mathbf{h}$($m$, $d$): 128 heuristic columns", size=P_SMALL, color=INK2, va="bottom")
    arrow(ax, XC + WC + 0.5, sy + sh / 2, XD - 1, sy + sh / 2)

    # ----------------------------------------------------------------- D: readout
    cx = XD + WD / 2
    # cross-fitting (top)
    yc = 9
    txt(ax, XD, yc, "Cross-fitting", bold=True, va="top")
    txt(ax, XD, yc + 3.6, "training pairs, K = 10 folds", size=P_SMALL, color=INK2, va="top")
    bx, by, bw, bh = XD, yc + 7.2, WD, 3.6
    K = 10
    for k in range(K):
        ax.add_patch(Rectangle((bx + k * bw / K, by), bw / K, bh, fc=INK2 if k == 3 else BLUE_T, ec="white", lw=0.4,
                               zorder=2))
    txt(ax, bx + 3.5 * bw / K, by + bh + 0.6, "query", size=P_SMALL, color=INK2, ha="center", va="top")
    txt(ax, bx + 7.5 * bw / K, by + bh + 0.6, "support", size=P_SMALL, color=BLUE, ha="center", va="top")
    txt(ax, XD, by + bh + 3.8, "a pair is never in\nits own support", size=P_SMALL, color=INK2, va="top")
    by2 = by + 17.5
    txt(ax, XD, by2 - 3.6, "test pairs", size=P_SMALL, color=INK2, va="top")
    ax.add_patch(Rectangle((bx, by2), bw, bh, fc=BLUE_T, ec="white", lw=0.4, zorder=2))
    txt(ax, cx, by2 + bh / 2, "support = all training pairs", size=P_SMALL, color=BLUE, ha="center", zorder=3)
    txt(ax, XD, by2 + bh + 1.0, "no heuristic reads\na test label", size=P_SMALL, color=INK2, va="top")
    line(ax, XD, 51, XD + WD, 51, color=HAIR, lw=0.4)
    # readout chain (bottom-up, fed by the h(m,d) strip)
    ymid = sy + sh / 2
    rbox(ax, XD, ymid - 4.5, WD, 9, fc=GREY_T, ec="none", r=0.8)
    txt(ax, cx, ymid - 1.5, "signed log1p", size=P_SMALL, ha="center")
    txt(ax, cx, ymid + 1.7, "z-score per column", size=P_SMALL, ha="center")
    arrow(ax, cx, ymid - 4.5, cx, ymid - 10.0)
    rbox(ax, XD, ymid - 23.0, WD, 13, fc="white", ec=INK2, lw=0.6, r=0.8)
    txt(ax, cx, ymid - 19.8, "logistic regression", size=P_SMALL, ha="center", color=INK2)
    txt(ax, cx, ymid - 16.1, "$\\sigma(\\mathbf{w}^{T}\\mathbf{z} + b)$", size=P_HEAD, ha="center")
    txt(ax, cx, ymid - 12.5, "≈130 parameters", size=P_SMALL, ha="center", color=INK2)
    arrow(ax, cx, ymid - 23.0, cx, ymid - 28.5)
    rbox(ax, XD + 3, ymid - 34.5, WD - 6, 6, fc=INK, ec="none", r=2.5)
    txt(ax, cx, ymid - 31.5, "score($m$, $d$)", bold=True, color="white", ha="center")

    fig.savefig(out + ".pdf")
    fig.savefig(out + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(HERE, "preview_Fig1.png"), dpi=200)
    plt.close(fig)
    print("wrote", out)


# ============================================================================= FIGURE 2
CATS = ["0", "1", "2–3", "4–7", "8–15", "16+"]
DEG_POS = {"MDR": [10.0, 10.4, 19.9, 29.2, 26.8, 3.7], "MDS": [10.7, 11.5, 17.7, 28.9, 28.1, 3.1]}
DEG_LOSS = {"MDR": [13.4, 19.6, 22.5, 26.6, 15.8, 2.1], "MDS": [12.3, 22.3, 22.3, 25.6, 16.1, 1.4]}
GAIN_BASE = {"MDR": [0.054, 0.094, 0.055, 0.045, 0.030, 0.029], "MDS": [0.043, 0.088, 0.058, 0.041, 0.027, 0.022]}
GAIN_PHL = {"MDR": [0.050, 0.081, 0.047, 0.039, 0.027, 0.031], "MDS": [0.038, 0.076, 0.053, 0.036, 0.025, 0.023]}
XLAB = "miRNA degree in the training support graph"
GREY_BAR = "#B5B5B5"


def fig2(out):
    W, H = 178, 90
    fig = plt.figure(figsize=(W / MM, H / MM))
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.105, wspace=0.34, hspace=0.70)
    letters = np.array([["A", "B"], ["C", "D"]])
    x = np.arange(len(CATS))
    bw = 0.36
    for r, ds in enumerate(["MDR", "MDS"]):
        for c in range(2):
            ax = axes[r, c]
            if c == 0:
                s1, s2 = DEG_POS[ds], DEG_LOSS[ds]
                l1, l2 = "share of test positives", "share of ranking loss"
                ax.set_ylim(0, 33)
                ax.set_yticks([0, 10, 20, 30])
                ax.set_ylabel("Share (%)")
                loc = "upper left"
            else:
                s1, s2 = GAIN_BASE[ds], GAIN_PHL[ds]
                l1, l2 = "baseline (ctx + diff)", "PHL"
                ax.set_ylim(0, 0.10)
                ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
                ax.set_ylabel("Mean ranking loss per positive")
                loc = "upper right"
            ax.bar(x - bw / 2, s1, bw, color=GREY_BAR, label=l1, zorder=2)
            ax.bar(x + bw / 2, s2, bw, color=BLUE, label=l2, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels(CATS)
            ax.set_xlabel(XLAB)
            ax.tick_params(labelsize=P_BODY, length=2, width=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=P_SMALL, loc=loc, handlelength=1.0, handleheight=0.9,
                      borderaxespad=0.2, labelspacing=0.3)
            bb = ax.get_position()
            fig.text(bb.x0 - 0.062, bb.y1 + 0.035, letters[r, c], fontsize=P_LETTER, fontweight="bold",
                     va="bottom", ha="left")
            fig.text(bb.x0, bb.y1 + 0.035, ds, fontsize=P_HEAD, fontweight="bold", va="bottom", ha="left")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(HERE, "preview_Fig2.png"), dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1", "all"):
        fig1(os.path.join(HERE, "Fig1_PHL_architecture"))
    if which in ("2", "all"):
        fig2(os.path.join(HERE, "Fig2_degree_diagnostics"))
