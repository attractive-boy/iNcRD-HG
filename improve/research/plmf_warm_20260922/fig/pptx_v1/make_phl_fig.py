# -*- coding: utf-8 -*-
"""Journal-format figures for the PHL manuscript (python-pptx -> LibreOffice -> PDF/TIFF).

Fig1_PHL_architecture   : method overview, 178 x 94 mm (double column)
Fig2_degree_diagnostics : degree-stratified loss diagnostics, 178 x 100 mm

Conventions (OUP Bioinformatics / BiB, Elsevier, Springer Nature all accept these):
  * artwork sized at final print width: 178 mm double column
  * white background, no title / caption / footnote / result numbers inside the artwork
  * Arial (Liberation Sans is the metric-compatible substitute when rendering on Linux)
  * text 6.5-8 pt at final size, panel letters 10 pt bold, lines 0.5-1 pt
  * English only; identity never rides on colour alone (every group also carries a text label)

Numbers in Fig 2 come from ../REPORT_WARM_PHL.md section 1.1 and 4.3 (error_analysis.py).
"""
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Mm, Pt

FONT = "Arial"
WHITE = "FFFFFF"
INK, INK2, MUTED = "000000", "404040", "707070"
GRID, HAIR, TILE = "D9D9D9", "B0B0B0", "F2F2F2"
BLUE, ORANGE, GREEN = "2A78D6", "E8652E", "159C6B"
BLUE_T, ORANGE_T, GREEN_T = "E3EEFA", "FCE9E0", "DDF2E9"
GREY_BAR = "B5B5B5"

# type scale (pt)
P_LETTER, P_HEAD, P_TITLE, P_BODY, P_SMALL = 10, 8, 7.5, 7, 6.5


def rgb(h):
    return RGBColor.from_string(h)


# ----------------------------------------------------------------- text helpers
def _set_ea(rPr, name):
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.Element(qn("a:ea"))
        if latin is not None:
            latin.addnext(ea)
        else:
            rPr.append(ea)
    ea.set("typeface", name)


def set_font(run, size, bold=False, color=INK, italic=False, name=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = rgb(color)
    f.name = name
    _set_ea(run._r.get_or_add_rPr(), name)


def fill_text(tf, lines, size=P_BODY, color=INK, bold=False, align=PP_ALIGN.LEFT,
              anchor=MSO_ANCHOR.TOP, margins=(1.0, 0.6, 1.0, 0.6), space_after=1.0,
              line_spacing=1.0, wrap=True):
    """lines: list of str | (text, kw) | list-of-(text, kw). margins in mm."""
    tf.word_wrap = wrap
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = [Mm(m) for m in margins]
    tf.vertical_anchor = anchor
    first = True
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        if isinstance(line, str):
            runs = [(line, {})]
        elif isinstance(line, tuple) and len(line) == 2 and isinstance(line[1], dict):
            runs = [line]
        else:
            runs = line
        for text, kw in runs:
            r = p.add_run()
            r.text = text
            set_font(r, kw.get("size", size), kw.get("bold", bold), kw.get("color", color),
                     kw.get("italic", False))
    return tf


def add_text(slide, x, y, w, h, lines, **kw):
    tb = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
    fill_text(tb.text_frame, lines, **kw)
    return tb


# ---------------------------------------------------------------- shape helpers
def add_box(slide, x, y, w, h, fill=None, line=None, line_w=0.5, radius=0.08,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE, alpha=None, dash=None):
    shp = slide.shapes.add_shape(shape, Mm(x), Mm(y), Mm(w), Mm(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        shp.adjustments[0] = radius
    shp.shadow.inherit = False
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        if alpha is not None:
            clr = shp._element.spPr.find(qn("a:solidFill")).find(qn("a:srgbClr"))
            a = etree.SubElement(clr, qn("a:alpha"))
            a.set("val", str(int(alpha * 1000)))
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_w)
        if dash:
            ln = shp.line._get_or_add_ln()
            d = etree.SubElement(ln, qn("a:prstDash"))
            d.set("val", dash)
    else:
        shp.line.fill.background()
    return shp


def add_line(slide, x1, y1, x2, y2, color=INK2, width=0.75, head=False, dash=None,
             kind=MSO_CONNECTOR.STRAIGHT):
    c = slide.shapes.add_connector(kind, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        d = etree.SubElement(ln, qn("a:prstDash"))
        d.set("val", dash)
    if head:
        t = etree.SubElement(ln, qn("a:tailEnd"))
        t.set("type", "triangle")
        t.set("w", "med")
        t.set("len", "med")
    return c


def node_circle(slide, cx, cy, r, fill, line=None):
    return add_box(slide, cx - r, cy - r, 2 * r, 2 * r, fill=fill, line=line, shape=MSO_SHAPE.OVAL)


def node_square(slide, cx, cy, r, fill, line=None):
    return add_box(slide, cx - r, cy - r, 2 * r, 2 * r, fill=fill, line=line, radius=0.2)


def new_figure(width_mm, height_mm):
    prs = Presentation()
    prs.slide_width = Mm(width_mm)
    prs.slide_height = Mm(height_mm)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(WHITE)
    return prs, s


def strip_shadows(prs):
    """Remove theme shadow styles and any effect reference so no renderer draws a drop shadow."""
    import re
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    theme = prs.slide_masters[0].part.part_related_by(RT.THEME)
    xml = theme.blob.decode("utf-8")
    xml = re.sub(r"<a:effectStyleLst>.*?</a:effectStyleLst>",
                 "<a:effectStyleLst>" + "<a:effectStyle><a:effectLst/></a:effectStyle>" * 3 + "</a:effectStyleLst>",
                 xml, flags=re.S)
    theme._blob = xml.encode("utf-8")
    for slide in prs.slides:
        for el in slide.shapes._spTree.iter(qn("a:effectRef")):
            el.set("idx", "0")
        for el in slide.shapes._spTree.iter(qn("p:spPr")):
            if el.find(qn("a:effectLst")) is None:
                etree.SubElement(el, qn("a:effectLst"))


def save(prs, out):
    strip_shadows(prs)
    prs.save(out)
    print("saved", out)


def panel_label(slide, x, y, letter, heading=None, w=60):
    runs = [(letter, dict(size=P_LETTER, bold=True))]
    if heading:
        runs.append(("   " + heading, dict(size=P_HEAD, bold=True)))
    add_text(slide, x, y, w, 5, [runs], margins=(0, 0, 0, 0), anchor=MSO_ANCHOR.MIDDLE)


# =============================================================== FIGURE 1
ROWS = [
    dict(hue=BLUE, tint=BLUE_T,
         src=("Training positives", "miRNA–drug association matrix (MDR or MDS); positives of the training fold only"),
         g_title="Support graph + kNN content graph",
         g_cap=["Bipartite support graph from training positives (solid)",
                "kNN graph from miRNA / drug similarity (dashed)"],
         f_title="ctx (33) + diff (52)",
         f_body=["Degree, common neighbours, 3- and 5-step paths, rank-16 SVD, attribute support",
                 "PPR, Katz, 7-step paths with row / column normalisation"]),
    dict(hue=ORANGE, tint=ORANGE_T,
         src=("Curation provenance", "PMID of every association and the cell line / phenotype / disease context of the report"),
         g_title="Provenance hypergraph",
         g_cap=["Study hyperedge: all pairs reported by one PMID",
                "Context hyperedge: shared cell line / phenotype / disease",
                "Dashed: bridged pair (m₁, d₂)"],
         f_title="prov (24)",
         f_body=["PMID bridging |P(m) ∩ P(d)|, size-weighted, Jaccard (6)",
                 "Co-study transfer S(m)·A, A·S(d) (6)",
                 "Context-term overlap (4); record statistics (8)"]),
    dict(hue=GREEN, tint=GREEN_T,
         src=("Label-free miRNA knowledge", "HMDD v4 disease spectrum; miRNA name and seed family; degree in the other phenotype matrix"),
         g_title="miRNA side-information relations",
         g_cap=["miRNA → family / disease / cross-phenotype degree",
                "Adds information for zero- and low-degree miRNAs"],
         f_title="mprior (19)",
         f_body=["Name family (5); seed family (5)",
                 "HMDD disease bridge Σ_z H[m, z]·(HᵀA)[z, d] (7)",
                 "Cross-phenotype degree (2)"]),
]

# geometry (mm)
F1_W, F1_H = 178, 97
CA, WA = 2, 32
CB, WB = 38, 56
CC, WC = 98, 40
CD, WD = 142, 34
ROW_Y = [9, 38, 67]
ROW_H = 26


def draw_g1(s, px, py, hue):
    mx, dx = px + 7, px + 19
    ys = [py + 10, py + 15, py + 20]
    r = 1.2
    add_text(s, mx - 6, py + 5.5, 12, 3, ["miRNA"], size=P_SMALL, color=MUTED, align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0))
    add_text(s, dx - 6, py + 5.5, 12, 3, ["drug"], size=P_SMALL, color=MUTED, align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0))
    for (i, j) in [(0, 0), (0, 1), (1, 1), (2, 2)]:
        add_line(s, mx + r, ys[i], dx - r, ys[j], color=hue, width=0.9)
    add_line(s, mx, ys[1] + r, mx, ys[2] - r, color=INK2, width=0.75, dash="sysDash")
    add_line(s, dx, ys[0] + r, dx, ys[1] - r, color=INK2, width=0.75, dash="sysDash")
    for y in ys:
        node_circle(s, mx, y, r, fill=hue)
        node_square(s, dx, y, r, fill=WHITE, line=hue)


def draw_g2(s, px, py, hue):
    ox, oy, ow, oh = px + 2, py + 6.5, 22, 17
    add_box(s, ox, oy, ow, oh, fill=hue, alpha=15, shape=MSO_SHAPE.OVAL)
    add_text(s, ox - 1, oy + 0.6, ow + 2, 3, ["PMID hyperedge"], size=P_SMALL, color=INK2,
             align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0))
    mx, dx = px + 8.5, px + 17.5
    y1, y2 = py + 13.5, py + 19.5
    r = 1.2
    add_line(s, mx + r, y1, dx - r, y1, color=INK2, width=0.9)
    add_line(s, mx + r, y2, dx - r, y2, color=INK2, width=0.9)
    add_line(s, mx + r, y1 + 0.4, dx - r, y2 - 0.4, color=hue, width=1.1, dash="sysDash")
    node_circle(s, mx, y1, r, fill=hue)
    node_circle(s, mx, y2, r, fill=hue)
    node_square(s, dx, y1, r, fill=WHITE, line=hue)
    node_square(s, dx, y2, r, fill=WHITE, line=hue)
    for (x, y, t, al) in [(mx - 5.5, y1, "m₁", PP_ALIGN.RIGHT), (mx - 5.5, y2, "m₂", PP_ALIGN.RIGHT),
                          (dx + 1.6, y1, "d₁", PP_ALIGN.LEFT), (dx + 1.6, y2, "d₂", PP_ALIGN.LEFT)]:
        add_text(s, x, y - 1.5, 4, 3, [t], size=P_SMALL, color=INK2, align=al, margins=(0, 0, 0, 0))


def draw_g3(s, px, py, hue):
    cx, cy, r = px + 4.5, py + 15.5, 1.6
    sx, sw, sh = px + 9, 19, 3.8
    ys = [py + 8, py + 13.6, py + 19.2]
    labs = ["miRNA family", "HMDD diseases", "cross-phenotype"]
    for y, lab in zip(ys, labs):
        add_line(s, cx + r, cy, sx, y + sh / 2, color=hue, width=0.9)
        b = add_box(s, sx, y, sw, sh, fill=GREEN_T, line=None, radius=0.3)
        fill_text(b.text_frame, [lab], size=P_SMALL, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
                  margins=(0.3, 0, 0.3, 0), wrap=False)
    node_circle(s, cx, cy, r, fill=hue)
    add_text(s, px + 1, cy - r - 3.4, 7, 3, ["miRNA"], size=P_SMALL, color=MUTED,
             align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0), wrap=False)


def build_fig1(out):
    prs, s = new_figure(F1_W, F1_H)
    panel_label(s, CA, 2, "A", "Data sources", WA)
    panel_label(s, CB, 2, "B", "Relations (support-only)", WB)
    panel_label(s, CC, 2, "C", "Heuristic features", WC)
    panel_label(s, CD, 2, "D", "Linear readout", WD)

    strip_yc = ROW_Y[1] + 9.5
    for i, row in enumerate(ROWS):
        hue, tint = row["hue"], row["tint"]
        y = ROW_Y[i]
        yc = y + ROW_H / 2
        # A: data source
        b = add_box(s, CA, y, WA, ROW_H, fill=tint)
        fill_text(b.text_frame, [(row["src"][0], dict(bold=True, size=P_TITLE)), (row["src"][1], dict(size=P_BODY, color=INK2))],
                  anchor=MSO_ANCHOR.MIDDLE, margins=(2.0, 1.0, 1.6, 1.0), space_after=1.5)
        # B: relation panel
        add_box(s, CB, y, WB, ROW_H, fill=WHITE, line=hue, line_w=0.75)
        node_circle(s, CB + 2.6, y + 2.6, 0.9, fill=hue)
        add_text(s, CB + 4.2, y + 0.8, WB - 5, 3.8, [row["g_title"]], size=P_TITLE, bold=True, margins=(0, 0, 0, 0), anchor=MSO_ANCHOR.MIDDLE)
        [draw_g1, draw_g2, draw_g3][i](s, CB, y, hue)
        add_text(s, CB + 28.5, y + 5.5, WB - 29.5, ROW_H - 6, row["g_cap"], size=P_SMALL, color=INK2,
                 margins=(0.5, 0, 0.5, 0), space_after=1.2)
        # C: feature group
        add_box(s, CC, y, WC, ROW_H, fill=WHITE, line=HAIR, line_w=0.5, radius=0.05)
        add_box(s, CC, y + 2, 1.4, ROW_H - 4, fill=hue, shape=MSO_SHAPE.RECTANGLE)
        add_text(s, CC + 2.8, y + 1, WC - 3.8, ROW_H - 2,
                 [(row["f_title"], dict(bold=True, size=P_TITLE))] + [(t, dict(size=P_SMALL, color=INK2)) for t in row["f_body"]],
                 margins=(0, 0, 0, 0), space_after=1.5, anchor=MSO_ANCHOR.MIDDLE)
        # arrows A -> B -> C
        add_line(s, CA + WA, yc, CB, yc, head=True)
        add_line(s, CB + WB, yc, CC, yc, head=True)
        kind = MSO_CONNECTOR.STRAIGHT if i == 1 else MSO_CONNECTOR.ELBOW
        add_line(s, CC + WC, yc, CD, strip_yc, head=True, kind=kind)

    # D row 1: cross-fitting note
    b = add_box(s, CD, ROW_Y[0], WD, ROW_H, fill=TILE, line=None, radius=0.05)
    fill_text(b.text_frame, [("Leak-free cross-fitting", dict(bold=True, size=P_TITLE)),
                             ("Training rows: K = 10 inner folds; a pair never enters its own support", dict(size=P_SMALL, color=INK2)),
                             ("Test rows: all training positives as support", dict(size=P_SMALL, color=INK2))],
              margins=(1.6, 1.0, 1.4, 1.0), space_after=1.5, anchor=MSO_ANCHOR.MIDDLE)
    # D row 2: concatenation strip + legend
    y2 = ROW_Y[1]
    add_text(s, CD, y2 + 2.5, WD, 3.2, ["Concatenate 128 columns"], size=P_SMALL, color=INK2, align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0))
    sy, sh = strip_yc - 2.5, 5
    widths = [85, 24, 19]
    hues = [BLUE, ORANGE, GREEN]
    gap = 0.5
    usable = WD - 2 * gap
    x = CD
    for wv, hue in zip(widths, hues):
        w = usable * wv / 128
        add_box(s, x, sy, w, sh, fill=hue, shape=MSO_SHAPE.RECTANGLE)
        x += w + gap
    legend = ["ctx + diff (85)", "prov (24)", "mprior (19)"]
    ly = sy + sh + 1.2
    for k, (hue, lab) in enumerate(zip(hues, legend)):
        yy = ly + k * 3.0
        add_box(s, CD + 0.5, yy + 0.6, 2.0, 2.0, fill=hue, shape=MSO_SHAPE.RECTANGLE)
        add_text(s, CD + 3.2, yy, WD - 3.5, 3.2, [lab], size=P_SMALL, color=INK2, margins=(0, 0, 0, 0), anchor=MSO_ANCHOR.MIDDLE)
    # D row 3: readout
    y3 = ROW_Y[2]
    add_line(s, CD + WD / 2, ly + 9.2, CD + WD / 2, y3, head=True)
    b = add_box(s, CD, y3, WD, 16, fill=WHITE, line=INK2, line_w=0.75, radius=0.08)
    fill_text(b.text_frame, [("signed log1p → z-score", dict(size=P_SMALL, color=INK2)),
                             ("score(m, d) = wᵀz + b", dict(size=P_TITLE, bold=True)),
                             ("logistic regression", dict(size=P_SMALL, color=INK2)),
                             ("≈130 parameters", dict(size=P_SMALL, color=INK2))],
              align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margins=(1, 0.6, 1, 0.6), space_after=1.0)
    add_line(s, CD + WD / 2, y3 + 16, CD + WD / 2, y3 + 19.5, head=True)
    b = add_box(s, CD + 5, y3 + 19.5, WD - 10, 6, fill=INK, radius=0.5)
    fill_text(b.text_frame, [("score(m, d)", dict(size=P_TITLE, bold=True, color=WHITE))], align=PP_ALIGN.CENTER,
              anchor=MSO_ANCHOR.MIDDLE, margins=(0, 0, 0, 0))
    save(prs, out)


# =============================================================== FIGURE 2
CATS = ["0", "1", "2–3", "4–7", "8–15", "16+"]
DEG_POS = {"MDR": [10.0, 10.4, 19.9, 29.2, 26.8, 3.7], "MDS": [10.7, 11.5, 17.7, 28.9, 28.1, 3.1]}
DEG_LOSS = {"MDR": [13.4, 19.6, 22.5, 26.6, 15.8, 2.1], "MDS": [12.3, 22.3, 22.3, 25.6, 16.1, 1.4]}
GAIN_BASE = {"MDR": [0.054, 0.094, 0.055, 0.045, 0.030, 0.029], "MDS": [0.043, 0.088, 0.058, 0.041, 0.027, 0.022]}
GAIN_PHL = {"MDR": [0.050, 0.081, 0.047, 0.039, 0.027, 0.031], "MDS": [0.038, 0.076, 0.053, 0.036, 0.025, 0.023]}
XLAB = "miRNA degree in the training support graph"


def _patch_chart_fonts(chart, name=FONT):
    for tag in ("a:defRPr", "a:rPr", "a:endParaRPr"):
        for el in chart._chartSpace.iter(qn(tag)):
            latin = el.find(qn("a:latin"))
            if latin is None:
                latin = etree.SubElement(el, qn("a:latin"))
            latin.set("typeface", name)
            _set_ea(el, name)


def _chart_no_border(chart):
    cs = chart._chartSpace
    spPr = etree.Element(qn("c:spPr"))
    etree.SubElement(spPr, qn("a:noFill"))
    ln = etree.SubElement(spPr, qn("a:ln"))
    etree.SubElement(ln, qn("a:noFill"))
    cs.find(qn("c:chart")).addnext(spPr)


def _axis_title(axis, text):
    axis.has_title = True
    axis.axis_title.text_frame.text = text
    r = axis.axis_title.text_frame.paragraphs[0].runs[0]
    set_font(r, P_BODY, bold=False, color=INK)


def add_bar_chart(slide, x, y, w, h, series, colors, ymax, major, numfmt, ylab, xlab):
    cd = CategoryChartData()
    cd.categories = CATS
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Mm(x), Mm(y), Mm(w), Mm(h), cd)
    ch = gf.chart
    ch.font.size = Pt(P_BODY)
    ch.font.name = FONT
    ch.font.color.rgb = rgb(INK)
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(P_BODY)
    plot = ch.plots[0]
    plot.gap_width = 180
    plot.overlap = -10
    plot.vary_by_categories = False
    for sr, col in zip(plot.series, colors):
        sr.format.fill.solid()
        sr.format.fill.fore_color.rgb = rgb(col)
        sr.format.line.fill.background()
    va = ch.value_axis
    va.minimum_scale = 0
    va.maximum_scale = ymax
    va.major_unit = major
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(GRID)
    va.major_gridlines.format.line.width = Pt(0.5)
    va.format.line.color.rgb = rgb(INK)
    va.format.line.width = Pt(0.5)
    va.major_tick_mark = XL_TICK_MARK.OUTSIDE
    va.tick_labels.font.size = Pt(P_BODY)
    va.tick_labels.font.color.rgb = rgb(INK)
    va.tick_labels.number_format = numfmt
    va.tick_labels.number_format_is_linked = False
    _axis_title(va, ylab)
    ca = ch.category_axis
    ca.format.line.color.rgb = rgb(INK)
    ca.format.line.width = Pt(0.5)
    ca.major_tick_mark = XL_TICK_MARK.OUTSIDE
    ca.tick_labels.font.size = Pt(P_BODY)
    ca.tick_labels.font.color.rgb = rgb(INK)
    _axis_title(ca, xlab)
    _chart_no_border(ch)
    _patch_chart_fonts(ch)
    return ch


def build_fig2(out):
    W, H = 178, 100
    prs, s = new_figure(W, H)
    cw, chh = 85, 43
    cells = {"A": (2, 2), "B": (91, 2), "C": (2, 51), "D": (91, 51)}
    for letter, ds in [("A", "MDR"), ("B", "MDR"), ("C", "MDS"), ("D", "MDS")]:
        x, y = cells[letter]
        panel_label(s, x, y, letter, ds, 30)
        if letter in ("A", "C"):
            series = [("Share of test positives", DEG_POS[ds]), ("Share of ranking loss", DEG_LOSS[ds])]
            add_bar_chart(s, x, y + 4, cw, chh, series, [GREY_BAR, BLUE], 30, 10, '0"%"', "Share (%)", XLAB)
        else:
            series = [("Baseline (ctx + diff)", GAIN_BASE[ds]), ("PHL", GAIN_PHL[ds])]
            add_bar_chart(s, x, y + 4, cw, chh, series, [GREY_BAR, BLUE], 0.10, 0.02, "0.00", "Mean loss per positive", XLAB)
    save(prs, out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1", "all"):
        build_fig1("Fig1_PHL_architecture.pptx")
    if which in ("2", "all"):
        build_fig2("Fig2_degree_diagnostics.pptx")
