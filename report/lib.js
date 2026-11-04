// Допоміжні функції для генерації звіту (docx-js)
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak, VerticalAlign,
} = D;

const FONT = "Times New Roman";
const CW = 9355; // ширина текстової області, DXA (A4, поля 3 / 1,5 см)
const FIG_DIR = path.join(__dirname, "..", "figures");

const counters = { fig: 0, tab: 0 };

// **жирний**, `код`, [[плейсхолдер]] (жовте підсвічування)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  const push = (t, o = {}) => t && out.push(new TextRun({ text: t, font: FONT, ...base, ...o }));
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Courier New", size: (base.size || 24) - 2 });
    else push("[" + tok.slice(2, -2) + "]", { shading: { type: ShadingType.CLEAR, fill: "FFF2A8", color: "auto" } });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const P = (text, o = {}) =>
  new Paragraph({
    children: runs(text, o.run || {}),
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { line: o.line || 300, before: o.before ?? 0, after: o.after ?? 100 },
    indent: o.indent === undefined ? { firstLine: 567 } : o.indent,
    keepNext: o.keepNext,
    pageBreakBefore: o.pageBreakBefore,
  });

const PN = (text, o = {}) => P(text, { indent: { firstLine: 0 }, ...o }); // без відступу першого рядка
const Center = (text, o = {}) => P(text, { align: AlignmentType.CENTER, indent: { firstLine: 0 }, ...o });
const H1 = (t, pb = true) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: pb });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })], keepNext: true });

const Bullet = (text, level = 0) =>
  new Paragraph({
    children: runs(text),
    numbering: { reference: "bullets", level },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 300, after: 60 },
  });
const Numbered = (text, ref = "nums") =>
  new Paragraph({ children: runs(text), numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { line: 300, after: 60 } });

const Formula = (text) =>
  new Paragraph({
    children: [new TextRun({ text, font: "Cambria Math", size: 22, italics: true })],
    alignment: AlignmentType.LEFT,
    spacing: { before: 40, after: 100, line: 280 },
    indent: { left: 340, right: 200 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "7F9BBF", space: 8 } },
  });

const Caption = (text, kind) => {
  counters[kind] += 1;
  const label = kind === "fig" ? "Рисунок" : "Таблиця";
  return new Paragraph({
    children: [new TextRun({ text: `${label} ${counters[kind]} — ${text}`, font: FONT, size: 22, italics: kind === "fig" })],
    alignment: kind === "fig" ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: kind === "fig" ? 60 : 160, after: kind === "fig" ? 200 : 60 },
    keepNext: kind === "tab",
  });
};

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function Figure(name, caption, widthPx = 610) {
  const file = path.join(FIG_DIR, name);
  const { w, h } = pngSize(file);
  const height = Math.round((widthPx * h) / w);
  return [
    new Paragraph({
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height }, altText: { title: caption, description: caption, name } })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 0 },
      keepNext: true,
    }),
    Caption(caption, "fig"),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(content, width, o = {}) {
  const arr = Array.isArray(content) ? content : [content];
  const size = o.size || 19;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.TOP,
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    columnSpan: o.span,
    children: arr.map(
      (t) =>
        new Paragraph({
          children: runs(String(t), { size, bold: o.bold }),
          alignment: o.align || AlignmentType.LEFT,
          spacing: { line: 240, after: 20 },
        })
    ),
  });
}

// headers: string[]; rows: (string|string[])[][]; widths sum = CW
function Tab(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((h, i) => cell(h, widths[i], { fill: "DCE6F2", bold: true, size: o.size || 19, align: AlignmentType.CENTER })),
  });
  const body = rows.map(
    (r, ri) =>
      new TableRow({
        cantSplit: o.cantSplit ?? true,
        children: r.map((c, i) =>
          cell(c, widths[i], { size: o.size || 19, align: o.aligns ? o.aligns[i] : AlignmentType.LEFT, fill: o.zebra && ri % 2 ? "F5F8FC" : undefined })
        ),
      })
  );
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

const Gap = (n = 120) => new Paragraph({ children: [], spacing: { after: n } });
const Break = () => new Paragraph({ children: [new PageBreak()] });

// Лістинг коду: кожен рядок — окремий абзац (Courier New 6,5 pt)
function Code(file) {
  const src = fs.readFileSync(file, "utf-8").replace(/\t/g, "    ").split("\n");
  return src.map(
    (line) =>
      new Paragraph({
        children: [new TextRun({ text: line.length ? line : " ", font: "Courier New", size: 13 })],
        spacing: { line: 200, after: 0 },
        alignment: AlignmentType.LEFT,
      })
  );
}

module.exports = { D, FONT, CW, P, PN, Center, H1, H2, H3, Bullet, Numbered, Formula, Caption, Figure, Tab, Gap, Break, Code, runs, cell, counters };
