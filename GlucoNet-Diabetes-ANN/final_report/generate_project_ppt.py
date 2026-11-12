"""Build the full project PowerPoint (widescreen) for the ANN diabetes report.

Matches the Review I Keynote palette and layout so the two decks feel like one
project. Covers every required report chapter: abstract-level framing, literature
comparison table, methodology with block schematic and equations, experimental
results with output screenshots, ablation / SOTA, and conclusion.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
LOGO = ROOT / "assets" / "srm_logo.jpeg"
OUTPUT = ROOT / "Diabetes_Prediction_ANN_Project_Presentation.pptx"

NAVY = RGBColor(0x0E, 0x2A, 0x47)
NAVY_MID = RGBColor(0x1A, 0x4A, 0x73)
TEAL = RGBColor(0x1F, 0x6F, 0x8B)
ACCENT = RGBColor(0xC4, 0x5C, 0x26)
GOLD = RGBColor(0xC9, 0xA2, 0x27)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFFWHITE = RGBColor(0xF6, 0xF3, 0xEE)
LIGHT = RGBColor(0xE8, 0xEE, 0xF4)
BODY = RGBColor(0x1F, 0x2A, 0x37)
MUTED = RGBColor(0x5B, 0x67, 0x75)
GREEN = RGBColor(0x1F, 0x6B, 0x4A)
RED = RGBColor(0xA3, 0x2D, 0x2D)
AMBER_BG = RGBColor(0xFB, 0xF0, 0xD9)
AMBER = RGBColor(0x7A, 0x52, 0x00)
HI = RGBColor(0xFD, 0xE8, 0xD0)
ORANGE = RGBColor(0xC4, 0x5C, 0x26)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TOTAL = 16


def _set_run(run, size=16, bold=False, color=BODY, name="Calibri", italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = name


def _rect(slide, l, t, w, h, color, rounded=False):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    if rounded:
        try:
            sh.adjustments[0] = 0.08
        except Exception:
            pass
    return sh


def _line(slide, l, t, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def _box(slide, l, t, w, h, text, size=14, bold=False, color=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        tf._txBody.bodyPr.set("anchor", "t" if anchor == MSO_ANCHOR.TOP else "ctr")
    except Exception:
        pass
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size=size, bold=bold, color=color, italic=italic)
    return box


def _bullets(slide, l, t, w, h, items, size=15, color=BODY, spacing=8):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.space_after = Pt(spacing)
        run = p.add_run()
        run.text = "•  " + item
        _set_run(run, size=size, color=color)
    return box


def _add_table(slide, l, t, w, h, rows, col_widths=None, header=True,
               font_size=11, highlight=(), center_from=0):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), l, t, w, h)
    table = table_shape.table
    if col_widths:
        for i, width in enumerate(col_widths):
            table.columns[i].width = width
    hi = set(highlight)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            is_header = header and r == 0
            is_hi = r in hi
            p.alignment = (PP_ALIGN.CENTER if is_header or c >= center_from else PP_ALIGN.LEFT)
            run = p.add_run()
            run.text = str(value)
            _set_run(
                run,
                size=font_size if not is_header else font_size,
                bold=is_header or is_hi,
                color=WHITE if is_header else BODY,
            )
            fill = NAVY if is_header else (HI if is_hi else (LIGHT if r % 2 else WHITE))
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = tcPr.makeelement(qn(tag), {})
                srgb = ln.makeelement(qn("a:solidFill"), {})
                srgb.append(srgb.makeelement(qn("a:srgbClr"), {"val": "C5CDD6"}))
                ln.append(srgb)
                tcPr.append(ln)
    return table_shape


def _picture_fit(slide, path, left, top, max_w, max_h):
    path = Path(path)
    if not path.exists():
        return
    im = Image.open(path)
    w_px, h_px = im.size
    max_w_emu = int(max_w)
    max_h_emu = int(max_h)
    scale = min(max_w_emu / w_px, max_h_emu / h_px)
    nw = int(w_px * scale)
    nh = int(h_px * scale)
    l = int(left) + (max_w_emu - nw) // 2
    t = int(top) + (max_h_emu - nh) // 2
    slide.shapes.add_picture(str(path), Emu(l), Emu(t), Emu(nw), Emu(nh))


def _footer(slide, n, total=TOTAL):
    _line(slide, Inches(0), Inches(7.22), SLIDE_W, Inches(0.28), NAVY)
    _box(
        slide, Inches(0.35), Inches(7.24), Inches(10.4), Inches(0.24),
        "Educational model only — not a medical diagnostic tool.  Pima women, age 21+.",
        size=10, color=WHITE, anchor=MSO_ANCHOR.MIDDLE,
    )
    _box(
        slide, Inches(11.4), Inches(7.24), Inches(1.6), Inches(0.24),
        f"{n}  /  {total}", size=10, bold=True, color=WHITE,
        align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE,
    )


def _header(slide, title, subtitle=None):
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.92), NAVY)
    _line(slide, Inches(0), Inches(0.92), SLIDE_W, Inches(0.06), ACCENT)
    _box(slide, Inches(0.4), Inches(0.14), Inches(12.4), Inches(0.42),
         title, size=22, bold=True, color=WHITE)
    if subtitle:
        _box(slide, Inches(0.4), Inches(0.54), Inches(12.4), Inches(0.30),
             subtitle, size=13, color=RGBColor(0xD5, 0xDE, 0xE8))


def _blank():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def _slide(prs):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, OFFWHITE)
    return slide


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------


def slide_title(prs):
    slide = _slide(prs)
    _rect(slide, Inches(0), Inches(0), Inches(0.28), SLIDE_H, ACCENT)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.16), NAVY)
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(10.15), Inches(0.32), Inches(2.85), Inches(0.72))
    _box(slide, Inches(0.7), Inches(0.38), Inches(9.2), Inches(0.28),
         "21CSE326T  ·  ARTIFICIAL NEURAL NETWORKS  ·  OCTOBER 2026",
         size=13, bold=True, color=ACCENT)
    _box(slide, Inches(0.7), Inches(1.10), Inches(12.2), Inches(0.95),
         "Diabetes Prediction System", size=36, bold=True, color=NAVY)
    _box(slide, Inches(0.7), Inches(2.05), Inches(12.0), Inches(0.55),
         "A leakage-safe artificial neural network for binary diabetes-outcome classification",
         size=18, color=NAVY_MID)

    cards = [
        ("Dataset", "Pima Indians Diabetes\n768 rows  ·  8 predictors"),
        ("Model", "Feed-forward ANN\n8 → 16 → 8 → 1  ·  289 params"),
        ("Evaluation", "10 × 5 repeated CV\n+ 154-row hold-out"),
        ("Headline result", "ROC-AUC  0.834\nSensitivity  0.768"),
    ]
    for i, (title, body) in enumerate(cards):
        left = Inches(0.7 + i * 3.1)
        _rect(slide, left, Inches(2.80), Inches(2.95), Inches(1.55), WHITE, rounded=True)
        _line(slide, left, Inches(2.80), Inches(2.95), Inches(0.08), TEAL)
        _box(slide, left + Inches(0.15), Inches(2.96), Inches(2.65), Inches(0.32),
             title, size=13, bold=True, color=TEAL)
        _box(slide, left + Inches(0.15), Inches(3.30), Inches(2.65), Inches(0.90),
             body, size=14, color=BODY)

    _rect(slide, Inches(0.7), Inches(4.55), Inches(12.0), Inches(1.35), AMBER_BG, rounded=True)
    _box(slide, Inches(0.90), Inches(4.68), Inches(11.6), Inches(1.10),
         "Educational model only — not a medical diagnostic tool.\n"
         "The table is limited to adult women (age 21+) of Pima heritage near Phoenix, Arizona. "
         "Results do not generalise to other populations and must not be used to diagnose diabetes.",
         size=14, color=AMBER)

    _box(slide, Inches(0.7), Inches(6.10), Inches(12.0), Inches(0.35),
         "Shanmugapriyan J  (RA2411026050011)     ·     Deepak V F  (RA2411026050015)",
         size=15, bold=True, color=NAVY)
    _box(slide, Inches(0.7), Inches(6.45), Inches(12.0), Inches(0.30),
         "B.Tech. CSE (AI & ML)  ·  III Year / V Semester  ·  Section A  ·  "
         "SRM Institute of Science and Technology, Tiruchirappalli",
         size=13, color=MUTED)
    _footer(slide, 1)


def slide_problem(prs):
    slide = _slide(prs)
    _header(slide, "1.  Problem, aim, and objectives",
            "Chapter 1  ·  binary classification on eight numeric measurements")
    _rect(slide, Inches(0.4), Inches(1.18), Inches(12.5), Inches(1.05), WHITE, rounded=True)
    _box(slide, Inches(0.55), Inches(1.24), Inches(12.2), Inches(0.24),
         "PROBLEM", size=11, bold=True, color=ACCENT)
    _box(slide, Inches(0.55), Inches(1.48), Inches(12.2), Inches(0.65),
         "Given eight measurements (pregnancies, 2-hour glucose, blood pressure, skinfold, insulin, BMI, "
         "pedigree, age), classify each row as Outcome = 1 or 0. This is a classroom ANN task, not a clinic visit.",
         size=15, color=BODY)

    _rect(slide, Inches(0.4), Inches(2.38), Inches(6.15), Inches(1.20), WHITE, rounded=True)
    _box(slide, Inches(0.55), Inches(2.46), Inches(5.9), Inches(0.24), "AIM", size=11, bold=True, color=TEAL)
    _box(slide, Inches(0.55), Inches(2.72), Inches(5.9), Inches(0.75),
         "Train a small feed-forward ANN with a leakage-safe pipeline and compare it on identical folds with "
         "logistic regression and a random forest.",
         size=14, color=BODY)
    _rect(slide, Inches(6.70), Inches(2.38), Inches(6.20), Inches(1.20), WHITE, rounded=True)
    _box(slide, Inches(6.85), Inches(2.46), Inches(5.9), Inches(0.24),
         "RESEARCH QUESTIONS", size=11, bold=True, color=TEAL)
    _box(slide, Inches(6.85), Inches(2.72), Inches(5.9), Inches(0.75),
         "Does the ANN beat linear and tree baselines on sensitivity, F1, ROC-AUC, and PR-AUC? "
         "How much do protocol shortcuts inflate those numbers?",
         size=14, color=BODY)

    _box(slide, Inches(0.4), Inches(3.72), Inches(6.2), Inches(0.28),
         "Measurable objectives", size=15, bold=True, color=NAVY)
    _bullets(slide, Inches(0.4), Inches(4.02), Inches(6.3), Inches(2.95), [
        "Profile the local table: balance, zeros, duplicates",
        "Recode impossible zeros; keep Pregnancies = 0",
        "Fit impute / scale / oversample only on training folds",
        "Train an 8–16–8–1 MLP with Adam, L2, and early stopping",
        "Report the full metric set — not accuracy alone",
        "Ablate components and measure leakage optimism",
    ], size=14, spacing=5)

    _rect(slide, Inches(6.90), Inches(3.72), Inches(2.95), Inches(3.20), WHITE, rounded=True)
    _line(slide, Inches(6.90), Inches(3.72), Inches(2.95), Inches(0.08), GREEN)
    _box(slide, Inches(7.05), Inches(3.90), Inches(2.65), Inches(0.28),
         "IN SCOPE", size=12, bold=True, color=GREEN)
    _bullets(slide, Inches(7.00), Inches(4.24), Inches(2.75), Inches(2.50), [
        "768-row Pima table",
        "Small MLP ANN",
        "LR and forest",
        "Repeated CV + hold-out",
        "Ablation and leakage",
    ], size=13, spacing=5)

    _rect(slide, Inches(10.00), Inches(3.72), Inches(2.95), Inches(3.20), WHITE, rounded=True)
    _line(slide, Inches(10.00), Inches(3.72), Inches(2.95), Inches(0.08), RED)
    _box(slide, Inches(10.15), Inches(3.90), Inches(2.65), Inches(0.28),
         "OUT OF SCOPE", size=12, bold=True, color=RED)
    _bullets(slide, Inches(10.10), Inches(4.24), Inches(2.75), Inches(2.50), [
        "Medical diagnosis",
        "Other populations",
        "Deep / GPU stacks",
        "Test-set tuning",
        "External hospital data",
    ], size=13, spacing=5)
    _footer(slide, 2)


def slide_literature(prs):
    slide = _slide(prs)
    _header(slide, "2.  Literature survey — comparison of published work",
            "Chapter 2  ·  comparison table is mandatory  ·  numbers as reported by the authors")
    rows = [
        ["Work", "Technique", "Evaluation", "Reported result", "Limitation"],
        ["Smith et al. 1988", "ADAP (early neural)", "576 / 192 split",
         "Sens. = Spec. = 76%; ROC", "Single split; one community"],
        ["Temurtas et al. 2009", "MLNN-LM and PNN", "Split + 10-fold CV",
         "MLNN 82.37% accuracy", "Accuracy as the headline metric"],
        ["Sisodia & Sisodia 2018", "DT, SVM, naïve Bayes", "Cross-validation",
         "NB accuracy 76.30%", "No neural network compared"],
        ["Hasan et al. 2020", "MLP + weighted ensemble", "k-fold CV",
         "Ensemble AUC 0.950", "Aggressive cleaning; leakage risk"],
        ["Howlader et al. 2022", "Many selectors + GBR", "Cross-validation",
         "GBR accuracy 90.91%", "Far above typical 75–82% range"],
        ["This work", "Leakage-safe MLP 8–16–8–1", "10×5 repeated CV + hold-out",
         "ROC-AUC 0.834; Sens. 0.768", "Single population; no external set"],
    ]
    _add_table(
        slide, Inches(0.35), Inches(1.15), Inches(12.6), Inches(4.35), rows,
        col_widths=[Inches(2.35), Inches(2.45), Inches(2.35), Inches(2.75), Inches(2.70)],
        font_size=12, highlight=(6,),
    )
    _rect(slide, Inches(0.35), Inches(5.62), Inches(12.6), Inches(1.40), WHITE, rounded=True)
    _line(slide, Inches(0.35), Inches(5.62), Inches(0.12), Inches(1.40), ACCENT)
    _box(slide, Inches(0.65), Inches(5.70), Inches(12.1), Inches(0.26),
         "RESEARCH GAP", size=12, bold=True, color=ACCENT)
    _box(slide, Inches(0.65), Inches(5.98), Inches(12.1), Inches(0.90),
         "Many Pima papers leak preprocessing, mishandle zeros, and quote accuracy even though 65% of rows "
         "are class 0. This project answers that gap with medically justified zeros, fold-wise imputation / "
         "scaling / oversampling, same-fold LR and forest baselines, a full metric set, an ablation, and a "
         "measured leakage experiment.",
         size=14, color=BODY)
    _footer(slide, 3)


def slide_dataset(prs):
    slide = _slide(prs)
    _header(slide, "3.  Dataset and features",
            "Chapter 4.2  ·  Pima Indians Diabetes  ·  NIDDK / Smith et al. 1988  ·  local CSV only")
    _rect(slide, Inches(0.4), Inches(1.16), Inches(5.85), Inches(2.70), WHITE, rounded=True)
    _box(slide, Inches(0.55), Inches(1.26), Inches(5.55), Inches(0.26),
         "WHO IS IN THE TABLE", size=12, bold=True, color=TEAL)
    _bullets(slide, Inches(0.50), Inches(1.58), Inches(5.60), Inches(2.15), [
        "768 rows, 8 predictors + binary Outcome",
        "Adult women, age ≥ 21, Pima heritage",
        "NIDDK community near Phoenix, Arizona",
        "Local CSV / ARFF — code never downloads",
        "0 duplicates  ·  0 empty cells  ·  zeros are codes",
    ], size=14, spacing=5)

    rows = [
        ["Feature", "Meaning", "Zero?"],
        ["Pregnancies", "Times pregnant", "Valid — keep"],
        ["Glucose", "2-hour OGTT (mg/dL)", "Missing code"],
        ["BloodPressure", "Diastolic mm Hg", "Missing code"],
        ["SkinThickness", "Triceps skinfold mm", "Missing code"],
        ["Insulin", "2-hour serum insulin", "Missing code"],
        ["BMI", "kg / m²", "Missing code"],
        ["Pedigree (DPF)", "Family-history score", "Keep"],
        ["Age", "Years (minimum 21)", "Keep"],
    ]
    _add_table(
        slide, Inches(6.45), Inches(1.16), Inches(6.50), Inches(5.75), rows,
        col_widths=[Inches(2.05), Inches(2.55), Inches(1.90)], font_size=12,
    )
    _rect(slide, Inches(0.4), Inches(4.02), Inches(5.85), Inches(2.90), WHITE, rounded=True)
    _box(slide, Inches(0.55), Inches(4.12), Inches(5.55), Inches(0.26),
         "TARGET AND LIMITS", size=12, bold=True, color=TEAL)
    _bullets(slide, Inches(0.50), Inches(4.46), Inches(5.60), Inches(2.30), [
        "Outcome 1 = 2-hour glucose ≥ 200 mg/dL",
        "500 negative (65.1%) / 268 positive (34.9%)",
        "Dummy majority accuracy is already 65.1%",
        "One sex, one community, 1980s extract",
        "Cannot generalise to men, children, or other groups",
    ], size=14, spacing=5)
    _footer(slide, 4)


def slide_quality(prs):
    slide = _slide(prs)
    _header(slide, "4.  Data quality — computed from the local CSV",
            "768 rows  ·  0 duplicates  ·  0 true NaNs  ·  zeros in five lab columns are missing-value codes")
    stats = [
        ("500 / 268", "No diabetes / diabetes\nDummy accuracy ≈ 65%"),
        ("111", "Pregnancies = 0\nkept as valid"),
        ("374 / 227", "Insulin / Skin zeros\nrecoded as missing"),
        ("48.7%", "Insulin missing\nafter recode"),
    ]
    for i, (num, label) in enumerate(stats):
        left = Inches(0.4 + i * 3.2)
        _rect(slide, left, Inches(1.16), Inches(3.05), Inches(1.28), WHITE, rounded=True)
        _box(slide, left + Inches(0.12), Inches(1.22), Inches(2.80), Inches(0.42),
             num, size=20, bold=True, color=NAVY)
        _box(slide, left + Inches(0.12), Inches(1.64), Inches(2.80), Inches(0.70),
             label, size=12, color=MUTED)
    _picture_fit(slide, FIG / "fig_class_balance.png", Inches(0.35), Inches(2.58),
                 Inches(6.30), Inches(4.45))
    _picture_fit(slide, FIG / "fig_zero_counts.png", Inches(6.70), Inches(2.58),
                 Inches(6.25), Inches(4.45))
    _footer(slide, 5)


def slide_eda(prs):
    slide = _slide(prs)
    _header(slide, "5.  Exploratory data analysis",
            "Positive rows have higher glucose, BMI, and age  ·  Pearson r with Outcome after recoding zeros")
    _bullets(slide, Inches(0.4), Inches(1.12), Inches(12.5), Inches(0.85), [
        "Glucose separates the classes most clearly (median 107 vs 140 mg/dL). Insulin and skinfold only separate once zeros are recoded.",
        "Pearson r with Outcome: Glucose 0.49, BMI 0.31, Insulin 0.30, Age 0.24. Skin–BMI r ≈ 0.65; Glucose–Insulin r ≈ 0.58.",
    ], size=14, spacing=4)
    _picture_fit(slide, FIG / "fig_feature_distributions.png", Inches(0.30), Inches(2.05),
                 Inches(6.45), Inches(4.95))
    _picture_fit(slide, FIG / "fig_correlation.png", Inches(6.80), Inches(2.05),
                 Inches(6.15), Inches(4.95))
    _footer(slide, 6)


def slide_block(prs):
    slide = _slide(prs)
    _header(slide, "6.  Proposed methodology — block schematic",
            "Chapter 3  ·  Stage B is fitted on the training fold only  ·  the test fold is transformed, never refit")
    _picture_fit(slide, FIG / "fig_block_schematic.png", Inches(0.25), Inches(1.10),
                 Inches(12.80), Inches(5.95))
    _footer(slide, 7)


def slide_architecture(prs):
    slide = _slide(prs)
    _header(slide, "7.  ANN architecture and governing equations",
            "Multilayer perceptron  ·  8 → 16 → 8 → 1  ·  289 parameters  ·  ReLU hidden units  ·  sigmoid output")
    _picture_fit(slide, FIG / "fig_ann_architecture.png", Inches(0.25), Inches(1.12),
                 Inches(7.55), Inches(5.90))
    _rect(slide, Inches(7.90), Inches(1.16), Inches(5.05), Inches(5.85), WHITE, rounded=True)
    _box(slide, Inches(8.08), Inches(1.28), Inches(4.70), Inches(0.28),
         "FORWARD PASS", size=12, bold=True, color=TEAL)
    _box(slide, Inches(8.08), Inches(1.58), Inches(4.70), Inches(1.55),
         "a⁰ = x   (standardised inputs)\n"
         "zˡ = Wˡ aˡ⁻¹ + bˡ\n"
         "aˡ = g(zˡ),   g(z) = max(0, z)   ReLU\n"
         "ŷ  = σ(z³) = 1 / (1 + e⁻ᶻ³)",
         size=14, color=BODY)
    _box(slide, Inches(8.08), Inches(3.20), Inches(4.70), Inches(0.28),
         "TRAINING OBJECTIVE", size=12, bold=True, color=TEAL)
    _box(slide, Inches(8.08), Inches(3.50), Inches(4.70), Inches(1.55),
         "L = −[ y log ŷ + (1−y) log(1−ŷ) ]\n"
         "    + (α / 2) ||W||²\n"
         "Adam,  α = 10⁻³,  batch 32\n"
         "Early stop on a train-fold split",
         size=14, color=BODY)
    _box(slide, Inches(8.08), Inches(5.15), Inches(4.70), Inches(0.28),
         "DECISION RULE", size=12, bold=True, color=TEAL)
    _box(slide, Inches(8.08), Inches(5.45), Inches(4.70), Inches(1.30),
         "ŷ_class = 1  if  ŷ ≥ 0.5\n"
         "Parameters: (8·16+16)+(16·8+8)+(8·1+1)\n"
         "= 144 + 136 + 9 = 289",
         size=14, color=BODY)
    _footer(slide, 8)


def slide_pipeline(prs):
    slide = _slide(prs)
    _header(slide, "8.  Leakage-safe pipeline and training algorithm",
            "Every learned statistic comes from the training fold  ·  the test fold is scored once")
    steps = [
        ("1", "Zero → NaN", "Glucose, BP, skin,\ninsulin, BMI only"),
        ("2", "Median impute", "Medians learned\non the train fold"),
        ("3", "Standard scale", "μ and σ from the\ntraining fold"),
        ("4", "Oversample", "Random ROS of the\nminority, train only"),
        ("5", "Train MLP", "Adam + L2 +\nearly stopping"),
        ("6", "Score test", "Transform only\nno refit, no ROS"),
    ]
    for i, (num, title, body) in enumerate(steps):
        left = Inches(0.35 + i * 2.15)
        _rect(slide, left, Inches(1.18), Inches(2.05), Inches(2.35), WHITE, rounded=True)
        _rect(slide, left + Inches(0.12), Inches(1.32), Inches(0.38), Inches(0.38), NAVY, rounded=True)
        _box(slide, left + Inches(0.12), Inches(1.34), Inches(0.38), Inches(0.34),
             num, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, left + Inches(0.12), Inches(1.80), Inches(1.82), Inches(0.45),
             title, size=13, bold=True, color=NAVY)
        _box(slide, left + Inches(0.12), Inches(2.25), Inches(1.82), Inches(1.10),
             body, size=12, color=MUTED)
        if i < 5:
            _box(slide, left + Inches(1.92), Inches(2.05), Inches(0.28), Inches(0.35),
                 "→", size=16, bold=True, color=ACCENT)

    _rect(slide, Inches(0.4), Inches(3.70), Inches(7.55), Inches(3.25), WHITE, rounded=True)
    _box(slide, Inches(0.55), Inches(3.82), Inches(7.25), Inches(0.28),
         "ALGORITHM  —  STRATIFIED CROSS-VALIDATION", size=12, bold=True, color=TEAL)
    _box(slide, Inches(0.55), Inches(4.16), Inches(7.25), Inches(2.65),
         "1.  Split D into K = 5 stratified folds (seed 42).\n"
         "2.  For each fold k, let Train = D \\ Fold_k, Test = Fold_k.\n"
         "3.  Recode zeros; fit imputer and scaler on Train; apply to both.\n"
         "4.  Oversample Train only. Fit the MLP with early stopping on Train.\n"
         "5.  Transform Test with the fitted objects (never resample Test).\n"
         "6.  Predict and score. Repeat. Report mean ± sd over folds.\n"
         "7.  Repeat the 5-fold split 10 times for the headline estimate.",
         size=14, color=BODY)

    _rect(slide, Inches(8.15), Inches(3.70), Inches(4.75), Inches(3.25), AMBER_BG, rounded=True)
    _box(slide, Inches(8.30), Inches(3.82), Inches(4.45), Inches(0.28),
         "NOT ALLOWED", size=12, bold=True, color=AMBER)
    _bullets(slide, Inches(8.25), Inches(4.20), Inches(4.50), Inches(2.55), [
        "Fit scaler / imputer on all 768 rows",
        "Oversample the whole table, then CV",
        "Treat Pregnancies = 0 as missing",
        "Tune hidden size on the test fold",
        "Delete outliers using all rows",
    ], size=14, color=AMBER, spacing=6)
    _footer(slide, 9)


def slide_protocol(prs):
    slide = _slide(prs)
    _header(slide, "9.  Experimental setup",
            "Apple M5 CPU  ·  no GPU  ·  seed 42  ·  full suite ≈ 3 minutes  ·  four protocols, each for a stated purpose")
    cards = [
        ("P1  ·  seeded 5-fold", "Seed 42, five stratified folds.\n13 configurations compared.\nOOF ROC, PR, confusion."),
        ("P2  ·  10 × 5 repeated", "50 splits. Headline means.\nCorrected resampled t-tests\nvs logistic and forest."),
        ("P3  ·  3 × 5 repeated", "15 identical splits.\nAblation of components,\narchitecture, and leakage."),
        ("P4  ·  80 / 20 hold-out", "614 train / 154 test.\nChoices frozen first.\nSingle unseen test."),
    ]
    for i, (title, body) in enumerate(cards):
        left = Inches(0.4 + i * 3.2)
        _rect(slide, left, Inches(1.18), Inches(3.05), Inches(2.05), WHITE, rounded=True)
        _line(slide, left, Inches(1.18), Inches(3.05), Inches(0.08), TEAL)
        _box(slide, left + Inches(0.14), Inches(1.34), Inches(2.78), Inches(0.40),
             title, size=14, bold=True, color=NAVY)
        _box(slide, left + Inches(0.14), Inches(1.78), Inches(2.78), Inches(1.30),
             body, size=13, color=BODY)

    _picture_fit(slide, FIG / "fig_cv_protocol.png", Inches(0.35), Inches(3.40),
                 Inches(7.40), Inches(3.60))
    _rect(slide, Inches(7.95), Inches(3.40), Inches(4.95), Inches(3.60), WHITE, rounded=True)
    _box(slide, Inches(8.12), Inches(3.52), Inches(4.60), Inches(0.28),
         "MODELS ON IDENTICAL FOLDS", size=12, bold=True, color=TEAL)
    _bullets(slide, Inches(8.10), Inches(3.88), Inches(4.65), Inches(2.95), [
        "Dummy most-frequent (sanity floor)",
        "Logistic regression, class_weight balanced",
        "Random forest, 100 trees, depth 6",
        "MLP, no rebalancing (specificity-heavy)",
        "Proposed MLP + training-fold ROS",
        "Metrics: Acc, Prec, Sens, Spec, F1, ROC-AUC, PR-AUC",
    ], size=13, spacing=5)
    _footer(slide, 10)


def slide_results(prs):
    slide = _slide(prs)
    _header(slide, "10.  Experimental results — repeated cross-validation",
            "Protocol P2  ·  mean over 10 × 5 = 50 stratified splits  ·  proposed row highlighted")
    rows = [
        ["Model", "Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC", "PR-AUC"],
        ["Dummy (majority)", "0.651", "0.000", "1.000", "0.000", "0.500", "0.349"],
        ["Logistic regression", "0.754", "0.718", "0.774", "0.671", "0.836", "0.722"],
        ["Random forest", "0.761", "0.761", "0.761", "0.690", "0.838", "0.717"],
        ["MLP ANN (no ROS)", "0.760", "0.560", "0.867", "0.617", "0.830", "0.704"],
        ["Proposed MLP (+ROS)", "0.746", "0.768", "0.734", "0.679", "0.834", "0.711"],
    ]
    _add_table(
        slide, Inches(0.4), Inches(1.14), Inches(12.5), Inches(2.55), rows,
        col_widths=[Inches(2.70), Inches(1.63), Inches(1.63), Inches(1.63),
                    Inches(1.63), Inches(1.64), Inches(1.64)],
        font_size=13, highlight=(5,), center_from=1,
    )
    _picture_fit(slide, FIG / "fig_repeated_cv_bars.png", Inches(0.25), Inches(3.78),
                 Inches(8.55), Inches(3.25))
    _rect(slide, Inches(8.90), Inches(3.80), Inches(4.05), Inches(3.20), WHITE, rounded=True)
    _box(slide, Inches(9.05), Inches(3.92), Inches(3.75), Inches(0.28),
         "WHAT THE NUMBERS SAY", size=12, bold=True, color=TEAL)
    _box(slide, Inches(9.05), Inches(4.26), Inches(3.75), Inches(2.55),
         "Ranking (ROC-AUC) is statistically the same for LR, forest, and the ANN (p > 0.7).\n\n"
         "Training-fold oversampling raises ANN sensitivity by +0.208 (p < 0.001) — that is the operating-point choice, not extra signal.",
         size=13, color=BODY)
    _footer(slide, 11)


def slide_holdout(prs):
    slide = _slide(prs)
    _header(slide, "11.  Hold-out test and out-of-fold curves",
            "Protocol P4  ·  154 unseen rows  ·  proposed ANN found 43 of 54 diabetic cases  ·  ROC-AUC 0.821")
    stats = [
        ("43 / 54", "True positives\nmissed 11 cases"),
        ("72 / 100", "True negatives\n28 false alarms"),
        ("0.796", "Hold-out\nsensitivity"),
        ("0.821", "Hold-out\nROC-AUC"),
    ]
    for i, (num, label) in enumerate(stats):
        left = Inches(0.4 + i * 3.2)
        _rect(slide, left, Inches(1.16), Inches(3.05), Inches(1.15), WHITE, rounded=True)
        _box(slide, left + Inches(0.12), Inches(1.22), Inches(2.80), Inches(0.42),
             num, size=20, bold=True, color=NAVY)
        _box(slide, left + Inches(0.12), Inches(1.62), Inches(2.80), Inches(0.58),
             label, size=12, color=MUTED)
    _picture_fit(slide, FIG / "fig_holdout.png", Inches(0.25), Inches(2.42),
                 Inches(12.80), Inches(4.60))
    _footer(slide, 12)


def slide_screenshots(prs):
    slide = _slide(prs)
    _header(slide, "12.  Output screenshots from the experiment scripts",
            "Left: seeded 5-fold comparison of 13 configurations   ·   Right: hold-out predictions on unseen rows")
    _picture_fit(slide, FIG / "console_cv_main.png", Inches(0.25), Inches(1.12),
                 Inches(6.45), Inches(5.90))
    _picture_fit(slide, FIG / "console_predict_demo.png", Inches(6.75), Inches(1.12),
                 Inches(6.25), Inches(5.90))
    _footer(slide, 13)


def slide_ablation(prs):
    slide = _slide(prs)
    _header(slide, "13.  Ablation study",
            "Protocol P3  ·  one component changed at a time  ·  red = drop of more than 0.02 from the proposed system")
    _picture_fit(slide, FIG / "fig_ablation_components.png", Inches(0.20), Inches(1.10),
                 Inches(12.90), Inches(4.35))
    takeaways = [
        ("− scaling", "ROC-AUC 0.721\nnetwork ignores small-scale inputs"),
        ("− early stop", "ROC-AUC 0.784\n289 parameters over-fit 700 rows"),
        ("− oversample", "Sens. 0.553\nranking unchanged; threshold shifts"),
        ("− zero recode", "ROC-AUC 0.820\ncoded zeros look like real lows"),
    ]
    for i, (title, body) in enumerate(takeaways):
        left = Inches(0.4 + i * 3.2)
        _rect(slide, left, Inches(5.50), Inches(3.05), Inches(1.50), WHITE, rounded=True)
        _box(slide, left + Inches(0.12), Inches(5.58), Inches(2.80), Inches(0.30),
             title, size=13, bold=True, color=ACCENT)
        _box(slide, left + Inches(0.12), Inches(5.90), Inches(2.80), Inches(0.95),
             body, size=12, color=BODY)
    _footer(slide, 14)


def slide_leakage_sota(prs):
    slide = _slide(prs)
    _header(slide, "14.  Leakage experiment and state-of-the-art comparison",
            "Oversampling before CV inflates PR-AUC by ~0.12  ·  nested CV drops the tuned ROC-AUC from 0.835 to 0.819")
    _picture_fit(slide, FIG / "fig_leakage.png", Inches(0.20), Inches(1.10),
                 Inches(7.55), Inches(3.55))
    _rect(slide, Inches(7.85), Inches(1.14), Inches(5.10), Inches(3.50), WHITE, rounded=True)
    _box(slide, Inches(8.00), Inches(1.24), Inches(4.80), Inches(0.28),
         "LEAKAGE (ANN)", size=12, bold=True, color=ACCENT)
    _box(slide, Inches(8.00), Inches(1.56), Inches(4.80), Inches(2.90),
         "Safe protocol     F1 0.684   PR-AUC 0.706\n"
         "Oversample before CV\n"
         "                         F1 0.771   PR-AUC 0.825\n\n"
         "The gain is an artefact: copies of the same patient land in both folds. "
         "Impute/scale on all rows barely moves the score. Nested CV is 0.016 lower than the non-nested best.",
         size=13, color=BODY)

    rows = [
        ["Study", "Headline", "This work, same spirit"],
        ["Smith 1988", "Sens. = Spec. = 76%", "Sens. 0.768  ·  Spec. 0.734"],
        ["Temurtas 2009", "82.4% accuracy", "Accuracy 0.746 (not the goal)"],
        ["Sisodia 2018", "NB 76.3% accuracy", "ANN F1 0.679  ·  ROC-AUC 0.834"],
        ["Hasan 2020", "Ensemble AUC 0.950", "Leakage-safe AUC 0.834"],
        ["Howlader 2022", "GBR 90.9% accuracy", "We do not chase 90%+ on 768 rows"],
    ]
    _add_table(
        slide, Inches(0.35), Inches(4.75), Inches(12.60), Inches(2.25), rows,
        col_widths=[Inches(2.50), Inches(4.40), Inches(5.70)],
        font_size=12, highlight=(1,),
    )
    _footer(slide, 15)


def slide_conclusion(prs):
    slide = _slide(prs)
    _header(slide, "15.  Conclusion",
            "A small ANN is competitive with, but not superior to, classical models on this benchmark")
    findings = [
        ("1", "Honest ranking",
         "ROC-AUC 0.834 is statistically indistinguishable from logistic regression (0.836) and the random forest (0.838)."),
        ("2", "Operating point",
         "Training-fold oversampling raises sensitivity by 0.208 (p < 0.001) without changing ranking — the ANN’s practical advantage."),
        ("3", "What actually matters",
         "Standardisation, early stopping, and justified zero recoding. Depth, activation, L2, and SelectKBest do not."),
        ("4", "Protocol is the story",
         "Leaky oversampling inflates PR-AUC by 0.12. Nested CV costs ~0.02 of ROC-AUC. Seed spread is ~0.04."),
        ("5", "Clinical reading",
         "Glucose, age, BMI, pregnancies dominate permutation importance. The system is educational, not a diagnostic device."),
        ("6", "Limits",
         "768 rows, one community of adult women, 48.7% of insulin imputed, no external cohort."),
    ]
    for i, (num, title, body) in enumerate(findings):
        r, c = divmod(i, 3)
        left = Inches(0.4 + c * 4.25)
        top = Inches(1.16 + r * 2.35)
        _rect(slide, left, top, Inches(4.10), Inches(2.20), WHITE, rounded=True)
        _rect(slide, left + Inches(0.14), top + Inches(0.16), Inches(0.38), Inches(0.38), NAVY, rounded=True)
        _box(slide, left + Inches(0.14), top + Inches(0.18), Inches(0.38), Inches(0.34),
             num, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, left + Inches(0.62), top + Inches(0.18), Inches(3.30), Inches(0.36),
             title, size=15, bold=True, color=NAVY)
        _box(slide, left + Inches(0.16), top + Inches(0.68), Inches(3.78), Inches(1.35),
             body, size=13, color=BODY)
    _footer(slide, 16)


def main() -> None:
    prs = _blank()
    slide_title(prs)
    slide_problem(prs)
    slide_literature(prs)
    slide_dataset(prs)
    slide_quality(prs)
    slide_eda(prs)
    slide_block(prs)
    slide_architecture(prs)
    slide_pipeline(prs)
    slide_protocol(prs)
    slide_results(prs)
    slide_holdout(prs)
    slide_screenshots(prs)
    slide_ablation(prs)
    slide_leakage_sota(prs)
    slide_conclusion(prs)
    prs.save(OUTPUT)
    print(f"Wrote {OUTPUT}  ({TOTAL} slides)")


if __name__ == "__main__":
    main()
