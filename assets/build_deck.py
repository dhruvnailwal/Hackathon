"""Build the 4-slide pitch deck (16:9) for the Multi-Model Anomaly Detection Engine.

Slides:
  1. Problem Statement
  2. Proposed Solution
  3. Key Features
  4. Deliverables / Proposed Outcome

Charts live in assets/charts/; performance numbers are the deterministic eval
snapshot (results/current_baseline) — not illustrative placeholders.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# palette (matches assets/make_charts.py)
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x25, 0x37)
TEAL = RGBColor(0x1F, 0x8A, 0x8C)
AMBER = RGBColor(0xE0, 0xA0, 0x3D)
RED = RGBColor(0xC9, 0x4F, 0x4F)
GREEN = RGBColor(0x3D, 0x9C, 0x63)
GREY = RGBColor(0x9A, 0xA5, 0xB1)
LIGHT = RGBColor(0xF4, 0xF8, 0xFA)
CARD_BG = RGBColor(0xF4, 0xF8, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GREY = RGBColor(0x4A, 0x5A, 0x6A)
FONT = "Calibri"

CH = os.path.join(os.path.dirname(__file__), "charts")

W, H = Inches(13.333), Inches(7.5)


def new_deck():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


TOTAL_SLIDES = 5


def add_slide(prs, title, kicker, num):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # header band
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.05))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    band.shadow.inherit = False
    # accent strip
    acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.14), Inches(1.05))
    acc.fill.solid()
    acc.fill.fore_color.rgb = TEAL
    acc.line.fill.background()
    acc.shadow.inherit = False
    # title
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(9.3), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.name = FONT
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = WHITE
    # kicker
    ktb = slide.shapes.add_textbox(Inches(9.75), Inches(0.16), Inches(3.35), Inches(0.8))
    ktf = ktb.text_frame
    ktf.word_wrap = True
    kp = ktf.paragraphs[0]
    kp.alignment = PP_ALIGN.RIGHT
    kr = kp.add_run()
    kr.text = kicker
    kr.font.name = FONT
    kr.font.size = Pt(10.5)
    kr.font.color.rgb = RGBColor(0xBF, 0xD7, 0xE8)
    # footer
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(7.14), Inches(12.53), Pt(1.2))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xD5, 0xDD, 0xE4)
    line.line.fill.background()
    line.shadow.inherit = False
    ftb = slide.shapes.add_textbox(Inches(0.4), Inches(7.18), Inches(9.5), Inches(0.3))
    fp = ftb.text_frame.paragraphs[0]
    fr = fp.add_run()
    fr.text = "Multi-Model Anomaly Detection Engine  \u00b7  CDR \u00b7 Bank \u00b7 Social"
    fr.font.name = FONT
    fr.font.size = Pt(9)
    fr.font.color.rgb = GREY
    ntb = slide.shapes.add_textbox(Inches(12.3), Inches(7.18), Inches(0.8), Inches(0.3))
    np_ = ntb.text_frame.paragraphs[0]
    np_.alignment = PP_ALIGN.RIGHT
    nr = np_.add_run()
    nr.text = f"{num} / {TOTAL_SLIDES}"
    nr.font.name = FONT
    nr.font.size = Pt(9)
    nr.font.color.rgb = GREY
    return slide


def para(tf, text, size, color, bold=False, italic=False, align=PP_ALIGN.LEFT,
         first=False, space_after=2, font=FONT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return p


def card(slide, l, t, w, h, accent=NAVY, fill=CARD_BG, border=None, radius=0.055):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = border or accent
    shp.line.width = Pt(1.1)
    shp.shadow.inherit = False
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t + Inches(0.10), Inches(0.09), h - Inches(0.20))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    bar.shadow.inherit = False
    return shp


def textbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    return tf


def chip(slide, l, t, w, h, label, fill, txt_color=WHITE, size=10.5):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    shp.adjustments[0] = 0.5
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = False
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = txt_color
    return shp


def picture(slide, name, l, t, width=None, height=None):
    return slide.shapes.add_picture(os.path.join(CH, name), l, t, width=width, height=height)


# ===========================================================================
# SLIDE 1 — PROBLEM STATEMENT
# ===========================================================================
def slide_problem(prs):
    s = add_slide(prs, "1 \u00b7 Problem Statement",
                  "Financial crime is hidden across channels \u2014 single-source monitors can't see it", 1)

    # left: scale chart
    picture(s, "deck_aml_scale.png", Inches(0.35), Inches(1.32), width=Inches(6.3))
    # left: schema chaos chart
    picture(s, "deck_schema_chaos.png", Inches(0.35), Inches(3.80), width=Inches(6.3))

    # takeaway bar
    card(s, Inches(0.35), Inches(6.05), Inches(6.3), Inches(0.85), accent=RED, fill=RGBColor(0xFB, 0xE8, 0xE8))
    tf = textbox(s, Inches(0.62), Inches(6.14), Inches(5.9), Inches(0.7))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, "Consequence: no fixed parser can work. The pipeline must detect schema at "
            "runtime \u2014 and say exactly when a model cannot be supported.", 10.5, NAVY, bold=True, first=True)

    # right: three failure-mode cards
    fails = [
        ("Cross-source blindness", RED,
         "A call 19 minutes before a transfer is invisible to a bank-only monitor. "
         "The link between sources IS the crime signal."),
        ("Schema fragility", AMBER,
         "txn_date vs posted_at vs call_time \u2014 a hardcoded parser dies on the first "
         "new export format."),
        ("Coverage dishonesty", GREY,
         "Systems silently run with missing attributes \u2014 no \u201cinsufficient data\u201d "
         "story, no analyst trust, no accountability."),
    ]
    y = Inches(1.32)
    for title, color, body in fails:
        card(s, Inches(6.95), y, Inches(6.05), Inches(1.70), accent=color)
        tf = textbox(s, Inches(7.25), y + Inches(0.14), Inches(5.6), Inches(1.45))
        para(tf, title, 13, NAVY, bold=True, first=True, space_after=3)
        para(tf, body, 10.5, DARK_GREY)
        y += Inches(1.82)


# ===========================================================================
# SLIDE 2 — PROPOSED SOLUTION
# ===========================================================================
def slide_solution(prs):
    s = add_slide(prs, "2 \u00b7 Proposed Solution",
                  "One source-agnostic pipeline \u2014 only the models the data can support run", 2)

    # left: architecture
    picture(s, "architecture.png", Inches(0.30), Inches(1.30), width=Inches(7.15))

    # unified frame sample
    card(s, Inches(0.30), Inches(5.05), Inches(7.15), Inches(1.85), accent=TEAL)
    tf = textbox(s, Inches(0.52), Inches(5.14), Inches(6.8), Inches(0.3))
    para(tf, "Every model reads the same unified long-format frame", 11, NAVY, bold=True, first=True)
    rows = [
        ["entity_id", "event_type", "timestamp", "source", "amount", "counterparty"],
        ["E014", "call", "09:12", "cdr", "\u2014", "E027"],
        ["E014", "transaction", "09:24", "bank", "48,000", "E027"],
        ["E047", "post", "09:27", "social", "\u2014", "\u2014"],
    ]
    t = s.shapes.add_table(4, 6, Inches(0.55), Inches(5.48), Inches(6.65), Inches(1.3))
    t.table.rows[0].height = Inches(0.28)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.table.cell(ri, ci)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if ri == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT
            r.font.size = Pt(9.5)
            r.font.bold = ri == 0
            r.font.color.rgb = WHITE if ri == 0 else NAVY
    t.table.columns[0].width = Inches(0.9)
    t.table.columns[1].width = Inches(1.0)
    t.table.columns[2].width = Inches(1.0)
    t.table.columns[3].width = Inches(0.8)
    t.table.columns[4].width = Inches(1.1)
    t.table.columns[5].width = Inches(1.85)

    # right: five steps
    card(s, Inches(7.75), Inches(1.30), Inches(5.25), Inches(3.62), accent=NAVY)
    tf = textbox(s, Inches(8.05), Inches(1.42), Inches(4.75), Inches(3.4))
    para(tf, "The five-step pipeline", 13, NAVY, bold=True, first=True, space_after=4)
    steps = [
        ("Dynamic schema detection", "fuzzy header match + content probes; source type falls out of resolved fields"),
        ("Entity resolution", "ties one person across bank, call & social records without a shared ID"),
        ("Unified long-format frame", "entity \u00d7 event \u00d7 source \u00d7 amount \u00d7 counterparty"),
        ("Three analysis families in parallel", "time \u00b7 network \u00b7 statistical/ML outliers"),
        ("Fusion \u2192 explanation \u2192 dashboard", "which models fired, the evidence, the score"),
    ]
    for i, (h, b) in enumerate(steps):
        para(tf, f"{i+1}.  {h}", 11.5, TEAL if i % 2 else NAVY, bold=True, space_after=1)
        para(tf, b, 9.5, DARK_GREY, space_after=5)

    # sufficiency chips
    tf = textbox(s, Inches(7.75), Inches(5.05), Inches(5.25), Inches(0.3))
    para(tf, "Data-sufficiency engine \u2014 verdict per model, per source combination:", 10.5, NAVY, bold=True, first=True)
    chip(s, Inches(7.75), Inches(5.42), Inches(1.62), Inches(0.42), "SUPPORTED \u2192 run", GREEN)
    chip(s, Inches(9.47), Inches(5.42), Inches(1.72), Inches(0.42), "DEGRADED \u2192 run + warn", AMBER)
    chip(s, Inches(11.29), Inches(5.42), Inches(1.71), Inches(0.42), "BLOCKED \u2192 say why", RED)
    card(s, Inches(7.75), Inches(5.98), Inches(5.25), Inches(0.95), accent=AMBER, fill=RGBColor(0xFD, 0xEE, 0xDE))
    tf = textbox(s, Inches(8.05), Inches(6.06), Inches(4.75), Inches(0.8))
    para(tf, "\u201cNetwork-correlation model skipped: no counterparty/edge dimension "
            "resolved. Add a CDR (phone) or bank (account) file to enable it.\u201d",
         9.5, DARK_GREY, italic=True, first=True)


# ===========================================================================
# SLIDE 3 — KEY FEATURES
# ===========================================================================
def slide_features(prs):
    s = add_slide(prs, "3 \u00b7 Key Features",
                  "Six capabilities make the ensemble honest, explainable and provable", 3)

    feats = [
        ("Dynamic schema detection", "Fuzzy header match + content inference; handles renamed, unlabeled and mixed-format columns", TEAL),
        ("Entity resolution", "Blocking + weighted fuzzy linkage resolves one person across sources with no shared ID", TEAL),
        ("Multi-model ensemble", "Time \u00b7 network \u00b7 stat-ML \u00b7 Benford \u00b7 structuring \u00b7 behavioral \u00b7 chain fused into one ranking", TEAL),
        ("Data-sufficiency engine", "Per-model SUPPORTED / DEGRADED / BLOCKED verdicts with human-readable reasons \u2014 never silent failure", AMBER),
        ("Explainable insights", "Plain-language narration per flagged entity: which models fired, the evidence, the combined score", AMBER),
        ("One-command evaluation", "Recall/precision@k, per-type ablation, H1 check and surprise hold-out from a single command", AMBER),
    ]
    cw, ch, gx, gy = Inches(4.13), Inches(1.62), Inches(0.13), Inches(0.16)
    for i, (title, body, accent) in enumerate(feats):
        col, row = i % 3, i // 3
        l = Inches(0.35) + col * (cw + gx)
        t = Inches(1.30) + row * (ch + gy)
        card(s, l, t, cw, ch, accent=accent)
        tf = textbox(s, l + Inches(0.28), t + Inches(0.12), cw - Inches(0.44), ch - Inches(0.22))
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(tf, f"{i+1}", 16, accent, bold=True, first=True, space_after=2)
        para(tf, title, 12.5, NAVY, bold=True, space_after=2)
        para(tf, body, 9.8, DARK_GREY)

    picture(s, "activation_heatmap.png", Inches(0.30), Inches(4.72), width=Inches(6.35))
    picture(s, "deck_model_zoo.png", Inches(7.00), Inches(4.90), width=Inches(4.80))


# ===========================================================================
# SLIDE 4 — DELIVERABLES / PROPOSED OUTCOME (targets only, no measured)
# ===========================================================================
def slide_deliverables(prs):
    s = add_slide(prs, "4 \u00b7 Deliverables / Proposed Outcome",
                  "What we ship \u2014 and the targets that prove it", 4)

    # left: deliverables checklist
    card(s, Inches(0.30), Inches(1.30), Inches(5.55), Inches(5.05), accent=GREEN)
    tf = textbox(s, Inches(0.58), Inches(1.44), Inches(5.0), Inches(4.8))
    para(tf, "Deliverables", 13, NAVY, bold=True, first=True, space_after=5)
    dels = [
        ("Auto-adaptive ingestion + schema detection", "runs on any export format"),
        ("Capability registry + sufficiency engine", "SUPPORTED / DEGRADED / BLOCKED"),
        ("Multi-model ensemble", "7 model families on one shared dataframe"),
        ("Fusion ranking + explanation generator", "Borda + learned meta-learner"),
        ("GUI dashboard", "drag & drop, history, JSON/CSV/prose reports"),
        ("Synthetic data + surprise answer key", "write-only, leak-proof ground truth"),
        ("One-command evaluation harness", "every claim re-producible"),
    ]
    for title, sub in dels:
        para(tf, "\u2713  " + title, 11, NAVY, bold=True, space_after=0)
        para(tf, "      " + sub, 9.5, DARK_GREY, space_after=5)

    # left bottom: impact card
    card(s, Inches(0.30), Inches(6.45), Inches(5.55), Inches(0.60), accent=NAVY, fill=RGBColor(0xEA, 0xF3, 0xEA))
    tf = textbox(s, Inches(0.58), Inches(6.51), Inches(5.0), Inches(0.5))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, "Expected impact: fewer false queues, higher analyst trust, adapts to any source mix.", 10, NAVY, bold=True, first=True)

    # right top: targets chart
    picture(s, "deck_targets.png", Inches(6.15), Inches(1.30), width=Inches(6.85))

    # right bottom: targets table (how each is proven)
    rows = [
        ["Metric", "Target", "How it is proven"],
        ["recall@10 \u00b7 predefined storylines", "\u2265 0.95", "answer-key harness"],
        ["recall@10 \u00b7 surprise hold-out", "\u2265 0.80", "write-only surprise key"],
        ["precision@10 \u00b7 false-alert control", "\u2265 0.60", "harness + analyst review"],
        ["fusion vs best single model", "\u2265 +0.15", "leave-one-out ablation"],
        ["schema adoption \u00b7 renamed / unlabeled", "100%", "Stage-A stress test"],
        ["sufficiency verdicts", "exact model + reason", "scenario matrix"],
    ]
    t = s.shapes.add_table(len(rows), 3, Inches(6.15), Inches(3.92), Inches(6.85), Inches(2.55))
    t.table.columns[0].width = Inches(3.55)
    t.table.columns[1].width = Inches(1.35)
    t.table.columns[2].width = Inches(1.95)
    for ri, row in enumerate(rows):
        t.table.rows[ri].height = Inches(0.34)
        for ci, val in enumerate(row):
            cell = t.table.cell(ri, ci)
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if ri == 0 else (RGBColor(0xED, 0xF3, 0xF6) if ri % 2 else WHITE)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci != 1 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = FONT
            r.font.size = Pt(10)
            r.font.bold = ri == 0 or ci == 0 or ci == 1
            r.font.color.rgb = WHITE if ri == 0 else NAVY

    # right bottom: generalization strip
    card(s, Inches(6.15), Inches(6.62), Inches(6.85), Inches(0.48), accent=TEAL, fill=RGBColor(0xEA, 0xF3, 0xEA))
    tf = textbox(s, Inches(6.40), Inches(6.67), Inches(6.4), Inches(0.38))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, "Generalizes beyond banking: e-commerce, ride-hailing, mobile-money \u2014 anywhere transaction + contact data overlaps.",
         9.5, NAVY, first=True)


# ===========================================================================
# SLIDE 5 — REAL-WORLD VALIDATION & TRUST (measured, not proposed)
# ===========================================================================
def slide_validation_trust(prs):
    s = add_slide(prs, "5 · Real-World Validation & Trust",
                  "Measured against external data, not just our own synthetic generator", 5)

    # left: real-world validation
    picture(s, "deck_real_world_validation.png", Inches(0.30), Inches(1.28), width=Inches(6.85))
    card(s, Inches(0.30), Inches(4.55), Inches(6.85), Inches(0.95), accent=TEAL, fill=RGBColor(0xEA, 0xF3, 0xEA))
    tf = textbox(s, Inches(0.55), Inches(4.63), Inches(6.4), Inches(0.8))
    para(tf, "Precision degrades gracefully, not collapses, as we make the test harder "
            "(1.00 → 0.80 @10 as prevalence drops 21.7% → 8.8%) — evidence of real ranking "
            "signal on SAML-D (Oztas et al., IEEE ICEBE 2023), never seen during development, "
            "not a lucky number at one enriched setting.",
         10, NAVY, bold=True, first=True)
    card(s, Inches(0.30), Inches(5.65), Inches(6.85), Inches(0.95), accent=AMBER, fill=RGBColor(0xFD, 0xEE, 0xDE))
    tf = textbox(s, Inches(0.55), Inches(5.73), Inches(6.4), Inches(0.8))
    para(tf, "Bonus: real-data validation surfaced two real bugs — a schema-detection gap "
            "that silently lost the entity dimension, and a crash-causing infinite recursion "
            "in the layering detector — both found and fixed.", 10, NAVY, first=True)

    # right: trust & compliance features
    card(s, Inches(7.35), Inches(1.28), Inches(5.65), Inches(2.55), accent=NAVY)
    tf = textbox(s, Inches(7.62), Inches(1.40), Inches(5.15), Inches(2.35))
    para(tf, "Legal-basis provenance", 13, NAVY, bold=True, first=True, space_after=4)
    para(tf, "Every run records who authorized the analysis and under what legal basis "
            "(warrant #, case ref, regulatory request) — printed directly into the exported "
            "report itself, not hidden metadata.", 10.5, DARK_GREY, space_after=8)
    para(tf, "“Legal basis / authorization: not recorded” shown honestly when left "
            "blank, instead of a fake default — the same never-hide-an-absence discipline "
            "as the sufficiency engine.", 9.8, DARK_GREY, italic=True)

    card(s, Inches(7.35), Inches(4.00), Inches(5.65), Inches(2.60), accent=TEAL)
    tf = textbox(s, Inches(7.62), Inches(4.12), Inches(5.15), Inches(2.40))
    para(tf, "Chain-of-custody sealing", 13, NAVY, bold=True, first=True, space_after=4)
    para(tf, "Every archived run is SHA-256 sealed file-by-file and chained to the previous "
            "run’s seal — an append-only ledger, live in the app.", 10.5, DARK_GREY, space_after=8)
    chip(s, Inches(7.62), Inches(5.55), Inches(2.55), Inches(0.42), "Verified: 18/18 match", GREEN, size=9.5)
    chip(s, Inches(10.30), Inches(5.55), Inches(2.55), Inches(0.42), "TAMPERED: report.md", RED, size=9.5)
    tf = textbox(s, Inches(7.62), Inches(6.05), Inches(5.15), Inches(0.5))
    para(tf, "Demoed live via the History dialog’s “Verify integrity” button.",
         9.5, DARK_GREY, italic=True, first=True)


def main():
    prs = new_deck()
    slide_problem(prs)
    slide_solution(prs)
    slide_features(prs)
    slide_deliverables(prs)
    slide_validation_trust(prs)
    out = os.path.join(os.path.dirname(__file__), "AML_Pitch_Deck_v2.pptx")
    prs.save(out)
    print("deck saved:", out)


if __name__ == "__main__":
    main()
