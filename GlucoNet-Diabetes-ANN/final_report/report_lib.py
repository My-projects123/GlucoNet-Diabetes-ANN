"""python-docx helpers for the project report.

Formatting rules from the submission brief: Times New Roman throughout,
body 12 pt, title 16 pt bold, headings 14 pt bold, single line spacing,
justified body text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402
from PIL import Image  # noqa: E402

FONT = "Times New Roman"
BODY_PT = 12
HEADING_PT = 14
TITLE_PT = 16
TABLE_PT = 11
TEXT_WIDTH_IN = 6.02  # A4 with 1.25" left and 1.0" right margins

JUSTIFY = WD_ALIGN_PARAGRAPH.JUSTIFY
CENTER = WD_ALIGN_PARAGRAPH.CENTER
LEFT = WD_ALIGN_PARAGRAPH.LEFT

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")

_HUMANIZE_MAP_PATH = Path(__file__).resolve().parent / "humanize_map.json"
try:
    _HUMANIZE_MAP = json.loads(_HUMANIZE_MAP_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    _HUMANIZE_MAP = {}


def _humanize(text: str) -> str:
    return _HUMANIZE_MAP.get(text, text)


def _set_rfonts(rpr) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)


def style_run(run, size=BODY_PT, bold=None, italic=None, color=None) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    _set_rfonts(run._r.get_or_add_rPr())


def add_runs(par, text: str, size=BODY_PT, bold=False, italic=False) -> None:
    for part in TOKEN.split(text):
        if not part:
            continue
        sub = sup = False
        b, i = bold, italic
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            part, i = part[1:-1], True
        elif part.startswith("_{"):
            part, sub = part[2:-1], True
        elif part.startswith("^{"):
            part, sup = part[2:-1], True
        run = par.add_run(part)
        style_run(run, size=size, bold=b, italic=i)
        if sub:
            run.font.subscript = True
        if sup:
            run.font.superscript = True


def _field_run(par, instr: str, placeholder: str = "1") -> None:
    run = par.add_run()
    style_run(run)
    for tag in ("begin", "instr", "separate", "text", "end"):
        if tag == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = instr
        elif tag == "text":
            el = OxmlElement("w:t")
            el.text = placeholder
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        run._r.append(el)


def _cell_borders(cell, **edges) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge, spec in edges.items():
        el = OxmlElement(f"w:{edge}")
        if spec is None:
            el.set(qn("w:val"), "nil")
        else:
            size, color = spec
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:color"), color)
        borders.append(el)


def _shade(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def _repeat_header(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def _set_table_widths(tbl, widths: list[float]) -> None:
    """Pages lays tables out from tblGrid/tblW, not from per-cell widths."""
    grid = tbl._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    tblpr = tbl._tbl.tblPr
    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:w"), str(int(sum(widths) * 1440)))
    tblw.set(qn("w:type"), "dxa")
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.insert_element_before(layout, "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription")
    for row in tbl.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)


def _no_split(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    el.set(qn("w:val"), "true")
    trpr.append(el)


class References:
    """IEEE-style numbering in order of first citation."""

    def __init__(self, catalogue: dict[str, str]):
        self.catalogue = catalogue
        self.order: list[str] = []

    def num(self, key: str) -> int:
        if key not in self.catalogue:
            raise KeyError(f"unknown reference {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def cite(self, *keys: str) -> str:
        nums = sorted({self.num(k) for k in keys})
        groups, start, prev = [], nums[0], nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            groups.append((start, prev))
            start = prev = n
        groups.append((start, prev))
        parts = [f"[{a}]" if a == b else (f"[{a}], [{b}]" if b == a + 1 else f"[{a}]–[{b}]") for a, b in groups]
        return ", ".join(parts)


class Report:
    def __init__(self, out_dir: Path, page_map: dict[str, int] | None = None, refs: References | None = None):
        self.out_dir = out_dir
        self.eq_dir = out_dir / "equations"
        self.eq_dir.mkdir(parents=True, exist_ok=True)
        self.page_map = page_map or {}
        self.refs = refs
        self.doc = Document()
        self.chapter = 0
        self.fig_count = 0
        self.tab_count = 0
        self.eq_count = 0
        self.alg_count = 0
        self.headings: list[tuple[int, str]] = []
        self.figures: list[tuple[str, str]] = []
        self.tables: list[tuple[str, str]] = []
        self._setup_document()

    # ------------------------------------------------------------------ setup
    def _setup_document(self) -> None:
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
        sec.left_margin, sec.right_margin = Inches(1.25), Inches(1.0)
        sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(1.0)
        sec.footer_distance = Inches(0.5)

        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name = FONT
        normal.font.size = Pt(BODY_PT)
        _set_rfonts(normal.element.get_or_add_rPr())
        pf = normal.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.alignment = JUSTIFY

        for name, size, before, after in [("Heading 1", HEADING_PT, 0, 12), ("Heading 2", HEADING_PT, 12, 6)]:
            st = styles[name]
            st.font.name = FONT
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.italic = False
            st.font.color.rgb = RGBColor(0, 0, 0)
            _set_rfonts(st.element.get_or_add_rPr())
            st.paragraph_format.space_before = Pt(before)
            st.paragraph_format.space_after = Pt(after)
            st.paragraph_format.keep_with_next = True
            st.paragraph_format.line_spacing = 1.0
            st.paragraph_format.alignment = LEFT

        lb = styles["List Bullet"]
        lb.font.name = FONT
        lb.font.size = Pt(BODY_PT)
        _set_rfonts(lb.element.get_or_add_rPr())
        lb.paragraph_format.space_after = Pt(3)
        lb.paragraph_format.alignment = JUSTIFY

        # The default template draws bullets with the Symbol font; use a Times bullet instead.
        for lvl in self.doc.part.numbering_part.element.iter(qn("w:lvl")):
            rfonts = lvl.find(f"{qn('w:rPr')}/{qn('w:rFonts')}")
            if rfonts is not None and rfonts.get(qn("w:ascii")) == "Symbol":
                for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                    rfonts.set(qn(attr), FONT)
                lvl.find(qn("w:lvlText")).set(qn("w:val"), "•")

    # -------------------------------------------------------------- sections
    def new_numbered_section(self) -> None:
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        sec.footer.is_linked_to_previous = False
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:start"), "1")
        sec._sectPr.insert_element_before(
            pg, "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection",
            "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings", "w:sectPrChange")
        par = sec.footer.paragraphs[0]
        par.alignment = CENTER
        _field_run(par, " PAGE ")

    def page_break(self) -> None:
        par = self.doc.add_paragraph()
        par.add_run().add_break(WD_BREAK.PAGE)
        par.paragraph_format.space_after = Pt(0)

    # ---------------------------------------------------------------- text
    def para(self, text: str, align=JUSTIFY, size=BODY_PT, bold=False, italic=False, space_after=6,
             space_before=0, keep_with_next=False, indent=None):
        text = _humanize(text)
        par = self.doc.add_paragraph()
        par.alignment = align
        par.paragraph_format.space_after = Pt(space_after)
        par.paragraph_format.space_before = Pt(space_before)
        par.paragraph_format.keep_with_next = keep_with_next
        if indent is not None:
            par.paragraph_format.left_indent = Inches(indent)
        add_runs(par, text, size=size, bold=bold, italic=italic)
        return par

    def lead(self, label: str, text: str):
        """Paragraph with a bold run-in label (used instead of level-3 headings)."""
        return self.para(f"**{label}** {text}")

    def bullets(self, items: list[str]) -> None:
        for item in items:
            item = _humanize(item)
            par = self.doc.add_paragraph(style="List Bullet")
            par.alignment = JUSTIFY
            add_runs(par, item)

    def numbered(self, items: list[str], prefix: str = "") -> None:
        for k, item in enumerate(items, 1):
            item = _humanize(item)
            par = self.doc.add_paragraph()
            par.alignment = JUSTIFY
            par.paragraph_format.left_indent = Inches(0.45)
            par.paragraph_format.first_line_indent = Inches(-0.45)
            par.paragraph_format.space_after = Pt(3)
            par.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
            add_runs(par, f"{prefix}{k}.\t{item}")

    def chapter_heading(self, title: str, new_page: bool = True) -> None:
        self.chapter += 1
        self.fig_count = self.tab_count = self.eq_count = 0
        text = f"{self.chapter}. {title.upper()}"
        self.headings.append((1, text))
        h = self.doc.add_heading(text, level=1)
        h.paragraph_format.page_break_before = new_page

    def unnumbered_heading(self, title: str, toc: bool = True, center: bool = False,
                           new_page: bool = False) -> None:
        if toc:
            self.headings.append((1, title))
        h = self.doc.add_heading(title, level=1)
        h.paragraph_format.page_break_before = new_page
        if center:
            h.alignment = CENTER
        return h

    def section(self, title: str) -> None:
        self.headings.append((2, title))
        self.doc.add_heading(title, level=2)

    # ------------------------------------------------------------- figures
    def figure(self, path: Path, caption: str, width: float = 6.0, short: str | None = None) -> str:
        self.fig_count += 1
        label = f"Figure {self.chapter}.{self.fig_count}"
        par = self.doc.add_paragraph()
        par.alignment = CENTER
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_after = Pt(4)
        par.paragraph_format.space_before = Pt(6)
        par.add_run().add_picture(str(path), width=Inches(min(width, TEXT_WIDTH_IN)))
        cap = self.doc.add_paragraph()
        cap.alignment = CENTER
        cap.paragraph_format.space_after = Pt(10)
        run = cap.add_run(f"{label}: ")
        style_run(run, bold=True)
        add_runs(cap, caption)
        self.figures.append((label, short or caption))
        return label

    def next_figure_label(self, offset: int = 1) -> str:
        return f"Figure {self.chapter}.{self.fig_count + offset}"

    def next_table_label(self, offset: int = 1) -> str:
        return f"Table {self.chapter}.{self.tab_count + offset}"

    # -------------------------------------------------------------- tables
    def table(self, rows: list[list[str]], caption: str, widths: list[float] | None = None,
              size: float = TABLE_PT, align: list[str] | None = None, header_fill: str = "D9E2F3",
              bold_rows: tuple[int, ...] = (), short: str | None = None) -> str:
        self.tab_count += 1
        label = f"Table {self.chapter}.{self.tab_count}"
        cap = self.doc.add_paragraph()
        cap.alignment = CENTER
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.space_after = Pt(4)
        run = cap.add_run(f"{label}: ")
        style_run(run, bold=True)
        add_runs(cap, caption)
        self.tables.append((label, short or caption))

        ncols = len(rows[0])
        tbl = self.doc.add_table(rows=len(rows), cols=ncols)
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _set_table_widths(tbl, widths or [TEXT_WIDTH_IN / ncols] * ncols)
        for r, row in enumerate(rows):
            trow = tbl.rows[r]
            _no_split(trow)
            if r == 0:
                _repeat_header(trow)
            for c, value in enumerate(row):
                cell = trow.cells[c]
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.space_before = Pt(1)
                a = (align[c] if align else "l") if r else "c"
                par.alignment = {"l": LEFT, "c": CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT, "j": JUSTIFY}[a]
                add_runs(par, str(value), size=size, bold=(r == 0 or r in bold_rows))
                if r == 0:
                    _shade(cell, header_fill)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)
        return label

    # ----------------------------------------------------------- equations
    def equation(self, latex: str, fontsize: float = 13.0) -> str:
        self.eq_count += 1
        label = f"({self.chapter}.{self.eq_count})"
        path = self.eq_dir / f"eq_{self.chapter}_{self.eq_count}.png"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        plt.close(fig)
        with Image.open(path) as im:
            w_px, h_px = im.size
        width_in = min(w_px / 300.0, TEXT_WIDTH_IN - 0.8)
        par = self.doc.add_paragraph()
        par.alignment = LEFT
        par.paragraph_format.space_before = Pt(4)
        par.paragraph_format.space_after = Pt(6)
        tabs = par.paragraph_format.tab_stops
        tabs.add_tab_stop(Inches(TEXT_WIDTH_IN / 2), WD_TAB_ALIGNMENT.CENTER)
        tabs.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
        r = par.add_run("\t")
        style_run(r)
        par.add_run().add_picture(str(path), width=Inches(width_in))
        r = par.add_run(f"\t{label}")
        style_run(r)
        # Raise the number towards the vertical middle of the equation image. w:position is in
        # half-points, but Pages reads it as points; this value centres the number in the Pages PDF
        # and leaves it a little below centre in Word.
        height_pt = h_px / 300.0 * (width_in / (w_px / 300.0)) * 72
        pos = OxmlElement("w:position")
        pos.set(qn("w:val"), str(int(max(0.0, height_pt / 2 - 4))))
        r._r.get_or_add_rPr().find(qn("w:sz")).addprevious(pos)
        return label

    # ---------------------------------------------------------- algorithms
    def algorithm(self, title: str, lines: list[tuple[int, str]], io: list[str] | None = None) -> str:
        self.alg_count += 1
        label = f"Algorithm {self.alg_count}"
        # One unsplittable cell, so the title can never be separated from the steps.
        tbl = self.doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _set_table_widths(tbl, [TEXT_WIDTH_IN])
        heavy = (12, "000000")
        body = tbl.rows[0].cells[0]
        _cell_borders(body, top=heavy, bottom=heavy, left=None, right=None)
        _no_split(tbl.rows[0])
        hp = body.paragraphs[0]
        hp.alignment = LEFT
        hp.paragraph_format.space_after = Pt(3)
        pbdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "2"), ("w:color", "000000")):
            bottom.set(qn(k), v)
        pbdr.append(bottom)
        hp._p.get_or_add_pPr().insert_element_before(
            pbdr, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:spacing", "w:ind", "w:jc", "w:rPr")
        r = hp.add_run(f"{label}: ")
        style_run(r, size=TABLE_PT, bold=True)
        add_runs(hp, title, size=TABLE_PT, bold=True)
        for item in io or []:
            par = body.add_paragraph()
            par.paragraph_format.space_after = Pt(1)
            par.alignment = LEFT
            add_runs(par, item, size=TABLE_PT)
        for n, (indent, text) in enumerate(lines, 1):
            par = body.add_paragraph()
            par.alignment = LEFT
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.left_indent = Inches(0.35 + 0.28 * indent)
            par.paragraph_format.first_line_indent = Inches(-0.35 - 0.28 * indent)
            par.paragraph_format.tab_stops.add_tab_stop(Inches(0.35 + 0.28 * indent))
            add_runs(par, f"{n}:\t{text}", size=TABLE_PT)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return label

    # ------------------------------------------------------------- listings
    def anchor(self):
        par = self.doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        return par

    def fill_listing(self, anchor, entries: list[tuple[int, str, str]]) -> None:
        """Insert TOC-style lines (level, text, key) before the anchor paragraph."""
        for level, text, key in entries:
            par = anchor.insert_paragraph_before()
            par.alignment = LEFT
            par.paragraph_format.space_after = Pt(4 if level == 1 else 2)
            par.paragraph_format.left_indent = Inches(0.0 if level == 1 else 0.35)
            par.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT,
                                                          WD_TAB_LEADER.DOTS)
            page = self.page_map.get(key)
            page_txt = str(page) if page is not None else "00"
            add_runs(par, f"{text}\t{page_txt}", bold=(level == 1))

    def save(self, path: Path) -> None:
        self.doc.save(str(path))
