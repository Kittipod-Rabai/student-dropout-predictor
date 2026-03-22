import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── design tokens — warm gray palette ─────────────────── */
:root {
  --white:   #f5f0eb;
  --bg:      #e3dcd4;
  --surface: #ede8e3;
  --s2:      #ddd8d2;
  --s3:      #e3dcd4;

  --ink:     #1e1b18;
  --ink2:    #3b3632;
  --ink3:    #6b6560;
  --ink4:    #a8a09a;
  --ink5:    #cec8c2;

  --bdr:     rgba(60,40,20,.1);

  --green:  #1a6635; --gbg: #eef7f2; --gbdr: #9fd3b5; --gdk: #0d3d1f;
  --amber:  #7a4f00; --abg: #fdf6e8; --abdr: #e0be7a; --adk: #4a2e00;
  --red:    #7a1f1f; --rbg: #fdf0f0; --rbdr: #e8a0a0; --rdk: #4a0f0f;
  --blue:   #174ea6; --bbg: #eaf1fb; --bbdr: #9cbfe8;

  --r8:  8px;
  --r12: 12px;
  --r16: 16px;
  --r20: 20px;

  --ease: cubic-bezier(.4,0,.2,1);
}

/* ── base ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
               "Helvetica Neue", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] {
  background: rgba(245,240,235,.92) !important;
  backdrop-filter: blur(20px) !important;
  border-bottom: .5px solid rgba(60,40,20,.08) !important;
  box-shadow: none !important;
}
.block-container { padding: 32px 40px 80px !important; max-width: 1080px !important; }

/* ── sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #f0ebe4 !important;
  border-right: .5px solid rgba(60,40,20,.08) !important;
  min-width: 220px !important; max-width: 220px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
  flex-direction: column !important; gap: 2px !important; padding: 0 8px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  display: flex !important; flex-direction: row !important;
  align-items: center !important; padding: 9px 12px !important;
  border-radius: var(--r12) !important; border: none !important;
  background: transparent !important; cursor: pointer !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--ink3) !important; transition: background .15s, color .15s !important;
  gap: 0 !important; margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: var(--s3) !important; color: var(--ink) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: var(--s3) !important; color: var(--ink) !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* ── widget labels ─────────────────────────────────────── */
[data-testid="stWidgetLabel"] > label,
[data-testid="stWidgetLabel"] p {
  font-size: 12px !important; font-weight: 500 !important;
  color: var(--ink3) !important; letter-spacing: .01em !important;
}

/* ── sliders ───────────────────────────────────────────── */
[data-testid="stSlider"] div[data-baseweb="slider"] > div { background: var(--s2) !important; }
[data-testid="stSlider"] [role="progressbar"],
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--ink) !important; }
[data-testid="stSlider"] [role="slider"] {
  background: var(--surface) !important; border: none !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.18), 0 0 0 .5px rgba(0,0,0,.08) !important;
  width: 18px !important; height: 18px !important; outline: none !important;
}
[data-testid="stThumbValue"] {
  font-size: 11px !important; font-weight: 700 !important;
  background: var(--ink) !important; color: var(--white) !important;
  border-radius: 6px !important; padding: 2px 7px !important;
  border: none !important;
}

/* ── radio accent ──────────────────────────────────────── */
div[data-baseweb="radio"] [role="radio"] div:first-child,
div[data-baseweb="radio"] [aria-checked="true"] div:first-child {
  background: var(--ink) !important; border-color: var(--ink) !important;
}

/* ── number input ──────────────────────────────────────── */
[data-testid="stNumberInput"] {
  border: .5px solid var(--bdr) !important; border-radius: var(--r8) !important;
  overflow: hidden !important; background: var(--surface) !important;
}
[data-testid="stNumberInput"] input {
  background: var(--surface) !important; border: none !important;
  font-size: 14px !important; font-weight: 600 !important;
  color: var(--ink) !important; text-align: center !important;
}
[data-testid="stNumberInput"] button {
  background: var(--surface) !important; border: none !important;
  border-left: .5px solid var(--bdr) !important; color: var(--ink3) !important;
}
[data-testid="stNumberInput"] button:hover { background: var(--s3) !important; }

/* ── selectbox ─────────────────────────────────────────── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--surface) !important; border: .5px solid var(--bdr) !important;
  border-radius: var(--r8) !important;
  font-size: 13px !important; font-weight: 500 !important; color: var(--ink) !important;
}

/* ── segmented control ─────────────────────────────────── */
.seg-wrap [data-testid="stRadio"] > label { display: none !important; }
.seg-wrap [data-testid="stRadio"] div[role="radiogroup"] {
  flex-direction: row !important; background: var(--s3) !important;
  border-radius: var(--r8) !important; padding: 2px !important; gap: 1px !important;
  border: .5px solid var(--bdr) !important;
}
.seg-wrap [data-testid="stRadio"] label {
  flex: 1 !important; justify-content: center !important;
  padding: 6px 4px !important; border-radius: 7px !important;
  font-size: 12px !important; color: var(--ink3) !important;
  background: none !important; font-weight: 500 !important; margin: 0 !important;
}
.seg-wrap [data-testid="stRadio"] label:has(input:checked) {
  background: var(--surface) !important; color: var(--ink) !important;
  font-weight: 600 !important; box-shadow: 0 1px 3px rgba(0,0,0,.1) !important;
}

/* ── primary button ────────────────────────────────────── */
[data-testid="stButton"] > button {
  width: 100% !important; padding: 14px 0 !important;
  background: var(--ink) !important; color: var(--white) !important;
  border: none !important; border-radius: var(--r12) !important;
  font-size: 15px !important; font-weight: 600 !important;
  letter-spacing: -.01em !important; transition: all .15s var(--ease) !important;
}
[data-testid="stButton"] > button:hover { background: #3a3a3c !important; }
[data-testid="stButton"] > button:active { transform: scale(.98) !important; }

/* ── alert ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--abg) !important; border: none !important;
  border-radius: var(--r8) !important; border-left: 3px solid var(--abdr) !important;
  font-size: 12px !important; color: var(--amber) !important;
}

/* ── progress ──────────────────────────────────────────── */
[data-testid="stProgress"] > div { background: var(--s2) !important; border-radius: 99px !important; }
[data-testid="stProgress"] > div > div { background: var(--ink) !important; border-radius: 99px !important; }

/* ── cards ─────────────────────────────────────────────── */
.fcard {
  background: var(--surface); border-radius: var(--r20);
  padding: 20px 22px; margin-bottom: 14px;
  border: .5px solid rgba(0,0,0,.06);
  animation: fadeUp .3s var(--ease) both;
}
.fcard-lbl {
  font-size: 10px; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: var(--ink5);
  margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.fcard-lbl::after { content: ''; flex: 1; height: .5px; background: var(--s3); }

/* ── validation banner ─────────────────────────────────── */
.val-banner {
  background: var(--abg); border: .5px solid var(--abdr);
  border-radius: var(--r12); padding: 10px 14px;
  font-size: 12px; color: var(--amber); display: flex;
  align-items: flex-start; gap: 8px; line-height: 1.5;
  margin-top: 4px;
}
.val-dot { width: 6px; height: 6px; border-radius: 50%;
  background: var(--amber); flex-shrink: 0; margin-top: 4px; }

/* ── submit wrapper ────────────────────────────────────── */
.submit-wrap { position: relative; }
.submit-locked [data-testid="stButton"] > button {
  background: var(--ink4) !important; cursor: not-allowed !important;
  pointer-events: none !important;
}

/* ── loading overlay ───────────────────────────────────── */
.loading-card {
  background: var(--surface); border-radius: var(--r20);
  border: .5px solid rgba(0,0,0,.06); padding: 32px 24px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 14px;
  animation: fadeUp .2s var(--ease) both;
}
.spin {
  width: 28px; height: 28px; border: 2px solid var(--s2);
  border-top-color: var(--ink); border-radius: 50%;
  animation: spin .7s linear infinite;
}
.loading-txt { font-size: 13px; font-weight: 500; color: var(--ink3); }

/* ── result cards ──────────────────────────────────────── */
.rh {
  border-radius: var(--r20); padding: 22px; margin-bottom: 12px;
  border: .5px solid transparent; animation: fadeUp .28s var(--ease) both;
}
.rh-idle { background: #f8f8fa; border-color: var(--s2); }
.rh-ok   { background: var(--gbg); border-color: var(--gbdr); }
.rh-warn { background: var(--abg); border-color: var(--abdr); }
.rh-bad  { background: var(--rbg); border-color: var(--rbdr); }

.ri { width: 40px; height: 40px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
.ri-idle { background: var(--s2); }
.ri-ok   { background: var(--green); }
.ri-warn { background: #c8860a; }
.ri-bad  { background: var(--red); }

.ey { font-size: 10px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; margin-bottom: 3px; }
.ey-idle { color: var(--ink5); }
.ey-ok   { color: var(--green); }
.ey-warn { color: var(--amber); }
.ey-bad  { color: var(--red); }

.rt { font-size: 19px; font-weight: 700; letter-spacing: -.02em;
  color: var(--ink); margin-bottom: 5px; }
.rb { font-size: 13px; color: var(--ink2); line-height: 1.6; }

/* confidence row with tooltip context */
.conf-row { display: flex; align-items: center; gap: 6px; margin-top: 14px; }
.cv { font-size: 16px; font-weight: 700; color: var(--ink); }
.cl { font-size: 11px; color: var(--ink4); }
.conf-ctx {
  font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500;
  border: .5px solid transparent;
}
.conf-hi  { background: var(--gbg); color: var(--green); border-color: var(--gbdr); }
.conf-md  { background: var(--abg); color: var(--amber); border-color: var(--abdr); }
.conf-lo  { background: var(--rbg); color: var(--red);   border-color: var(--rbdr); }

/* prob bars */
.pw {
  background: var(--surface); border: .5px solid var(--bdr);
  border-radius: var(--r16); padding: 16px; margin-bottom: 12px;
  animation: fadeUp .28s .05s var(--ease) both;
}
.pw-h { font-size: 10px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: var(--ink4); margin-bottom: 12px; }
.pb { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.pb:last-child { margin-bottom: 0; }
.pb-l { font-size: 12px; color: var(--ink3); width: 68px; flex-shrink: 0; }
.pb-t { flex: 1; height: 5px; background: var(--s2); border-radius: 3px; overflow: hidden; }
.pb-f { height: 100%; border-radius: 3px; transition: width .6s var(--ease); }
.pb-p { font-size: 12px; font-weight: 700; width: 34px; text-align: right; color: var(--ink2); }

/* signal list */
.sw {
  background: var(--surface); border: .5px solid var(--bdr);
  border-radius: var(--r16); padding: 16px; margin-bottom: 12px;
  animation: fadeUp .28s .1s var(--ease) both;
}
.sw-h { font-size: 10px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: var(--ink4); margin-bottom: 10px; }
.si { display: flex; align-items: flex-start; gap: 10px;
  padding: 9px 0; border-bottom: .5px solid var(--s3); }
.si:first-child { padding-top: 0; }
.si:last-child  { border-bottom: none; padding-bottom: 0; }
.sdw {
  width: 26px; height: 26px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.sdw-ok { background: var(--gbg); }
.sdw-warn { background: var(--abg); }
.sdw-bad  { background: var(--rbg); }
.sd { width: 8px; height: 8px; border-radius: 50%; }
.sd-ok   { background: var(--green); }
.sd-warn { background: #9a6200; }
.sd-bad  { background: var(--red); }
.st2 { font-size: 13px; font-weight: 500; color: var(--ink); line-height: 1.3; }
.ss  { font-size: 11px; color: var(--ink4); margin-top: 2px; }

/* chips */
.chip {
  display: inline-block; font-size: 11px; font-weight: 500;
  border-radius: 6px; padding: 3px 9px; margin: 2px 3px 0 0;
  border: .5px solid transparent;
}
.ch-b  { background: var(--bbg);  color: var(--blue);  border-color: var(--bbdr); }
.ch-g  { background: var(--gbg);  color: var(--green); border-color: var(--gbdr); }
.ch-a  { background: var(--abg);  color: var(--amber); border-color: var(--abdr); }
.ch-r  { background: var(--rbg);  color: var(--red);   border-color: var(--rbdr); }
.ch-gr { background: var(--s3);   color: var(--ink3);  border-color: var(--s2); }

/* note box */
.note-box {
  background: var(--abg); border: .5px solid var(--abdr);
  border-radius: var(--r12); padding: 12px 14px;
  font-size: 12px; color: var(--amber); line-height: 1.65; margin-top: 4px;
}
.fn { font-size: 11px; color: var(--ink4); text-align: center;
  line-height: 1.6; padding: 6px 0; }

/* page header */
.ph { margin-bottom: 22px; }
.ph h2 { font-size: 21px; font-weight: 700; letter-spacing: -.03em;
  color: var(--ink); margin: 0 0 3px; }
.ph p  { font-size: 13px; color: var(--ink3); margin: 0; }

/* mode badge (fallback vs ML) */
.mode-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 500; padding: 3px 9px;
  border-radius: 20px; border: .5px solid var(--bdr);
  background: var(--s3); color: var(--ink3); margin-bottom: 16px;
}
.mode-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
.mode-dot-fb { background: var(--amber); }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/pipeline.pkl")

try:
    model = load_model()
    MODEL_READY = True
except Exception:
    MODEL_READY = False

NAMES   = {0: "เรียนปกติดี",           1: "เริ่มมีสัญญาณเสี่ยง",  2: "เสี่ยงหยุดเรียนสูง"}
EYEBROW = {0: "สถานะ: เรียนปกติ",      1: "สถานะ: มีสัญญาณเสี่ยง", 2: "สถานะ: เสี่ยงสูง"}
ADVICE  = {
    0: "นักเรียนมีความตั้งใจดี ติดตามและให้กำลังใจต่อไป",
    1: "แนะนำให้ครูพูดคุยและสนับสนุนเพิ่มเติมภายใน 48 ชม.",
    2: "ความเสี่ยงสูง ควรติดต่อผู้ปกครองโดยเร็ว",
}
ST_MAP  = {0: "ok", 1: "warn", 2: "bad"}
ICONS   = {
    "idle": '<circle cx="9" cy="9" r="5.5" stroke="#c7c7cc" stroke-width="1.3"/>'
            '<path d="M9 6.5v3M9 11.5h.01" stroke="#c7c7cc" stroke-width="1.3" stroke-linecap="round"/>',
    "ok":   '<path d="M5 9.5l3 3 5.5-6" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
    "warn": '<path d="M9 5.5v4M9 12h.01" stroke="white" stroke-width="1.5" stroke-linecap="round"/>',
    "bad":  '<path d="M5.5 5.5l7 7M12.5 5.5l-7 7" stroke="white" stroke-width="1.5" stroke-linecap="round"/>',
}
BAR_CLR = ["#1a6635", "#9a6200", "#7a1f1f"]

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                padding:20px 16px 16px;border-bottom:.5px solid #e5e5ea;
                margin-bottom:8px">
      <div style="width:30px;height:30px;border-radius:8px;background:#1c1c1e;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <circle cx="7" cy="7" r="5" stroke="white" stroke-width="1.2"/>
          <path d="M4.5 7l2 2 3-4" stroke="white" stroke-width="1.2"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div style="font-size:13px;font-weight:700;color:#1c1c1e;letter-spacing:-.02em">
          Dropout Predictor</div>
        <div style="font-size:10px;color:#aeaeb2;margin-top:1px">Early Warning System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        " ",
        ["🔍  ประเมินนักเรียน", "📖  เกี่ยวกับ ML"],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_idle():
    st.markdown(f"""
    <div class="rh rh-idle">
      <div class="ri ri-idle">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">{ICONS["idle"]}</svg>
      </div>
      <div class="ey ey-idle">สถานะ</div>
      <div class="rt">ยังไม่ได้ประเมิน</div>
      <div class="rb">กรอกข้อมูลให้ครบแล้วกดปุ่มประเมิน</div>
    </div>
    """, unsafe_allow_html=True)


def conf_ctx_html(conf: int) -> str:
    if conf >= 75:
        return f'<span class="conf-ctx conf-hi">ความมั่นใจสูง</span>'
    elif conf >= 55:
        return f'<span class="conf-ctx conf-md">ความมั่นใจปานกลาง</span>'
    else:
        return f'<span class="conf-ctx conf-lo">ความมั่นใจต่ำ — ควรพิจารณาร่วมกับข้อมูลอื่น</span>'


def render_result(r: dict):
    pred, probs = r["pred"], r["probs"]
    st8  = ST_MAP[pred]
    conf = round(probs[pred] * 100)

    st.markdown(f"""
    <div class="rh rh-{st8}">
      <div class="ri ri-{st8}">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">{ICONS[st8]}</svg>
      </div>
      <div class="ey ey-{st8}">{EYEBROW[pred]}</div>
      <div class="rt">{NAMES[pred]}</div>
      <div class="rb">{ADVICE[pred]}</div>
      <div class="conf-row">
        <div class="cv">{conf}%</div>
        {conf_ctx_html(conf)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # probability bars
    bars = "".join(
        f'<div class="pb">'
        f'<div class="pb-l">{lbl}</div>'
        f'<div class="pb-t"><div class="pb-f" style="width:{round(probs[i]*100)}%;background:{BAR_CLR[i]}"></div></div>'
        f'<div class="pb-p">{round(probs[i]*100)}%</div></div>'
        for i, lbl in enumerate(["เรียนปกติ", "เริ่มเสี่ยง", "หยุดเรียน"])
    )
    st.markdown(
        f'<div class="pw"><div class="pw-h">โอกาสแต่ละสถานะ</div>{bars}</div>',
        unsafe_allow_html=True,
    )

    # signal list
    def mk(s, t, b):
        return (
            f'<div class="si">'
            f'<div class="sdw sdw-{s}"><div class="sd sd-{s}"></div></div>'
            f'<div><div class="st2">{t}</div><div class="ss">{b}</div></div>'
            f'</div>'
        )

    cr, la, lf, ds = r["cr"], r["la"], r["lf"], r["ds"]
    sigs = ""
    sigs += mk("ok",   "ส่งงานสม่ำเสมอ",    f"{round(cr*100)}% ของงานทั้งหมด") if cr >= .7 else \
            mk("warn", "ส่งงานไม่สม่ำเสมอ", f"{round(cr*100)}% — ควรติดตาม")   if cr >= .4 else \
            mk("bad",  "ส่งงานน้อยมาก",      f"{round(cr*100)}% — น่าเป็นห่วง")
    sigs += mk("ok",   f"เข้าระบบล่าสุด {la} วันก่อน", "ยังคง active")    if la <= 7  else \
            mk("warn", f"เข้าระบบล่าสุด {la} วันก่อน", "เริ่มห่างขึ้น")  if la <= 21 else \
            mk("bad",  f"ไม่ได้เข้าระบบมา {la} วัน",   "หายไปนาน")
    sigs += mk("ok",   f"เข้าระบบ {lf:.1f} ครั้ง/สัปดาห์", "สม่ำเสมอ")      if lf >= 5 else \
            mk("warn", f"เข้าระบบ {lf:.1f} ครั้ง/สัปดาห์", "เริ่มน้อยลง")  if lf >= 2 else \
            mk("bad",  "แทบไม่ได้เข้าระบบ",             f"{lf:.1f} ครั้ง/สัปดาห์")
    sigs += mk("ok",   "คะแนนเสี่ยงต่ำ",     f"ระดับ {ds:.2f} (เกณฑ์ < 0.35)") if ds < .35 else \
            mk("warn", "คะแนนเสี่ยงปานกลาง", f"ระดับ {ds:.2f} (เกณฑ์ 0.35–0.65)") if ds < .65 else \
            mk("bad",  "คะแนนเสี่ยงสูง",      f"ระดับ {ds:.2f} (เกณฑ์ > 0.65)")

    st.markdown(
        f'<div class="sw"><div class="sw-h">สัญญาณที่ตรวจพบ</div>{sigs}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fn">ผลนี้เป็นการประเมินเบื้องต้น<br>'
        'ควรใช้ร่วมกับดุลยพินิจของครูเสมอ</div>',
        unsafe_allow_html=True,
    )


def run_prediction(inputs: dict) -> dict:
    """Run model or fallback, always return result dict."""
    df_in = pd.DataFrame([inputs])
    if MODEL_READY:
        pred  = int(model.predict(df_in)[0])
        probs = model.predict_proba(df_in)[0].tolist()
    else:
        cr  = inputs["completion_rate"]
        la  = inputs["last_activity_days_ago"]
        lf  = inputs["login_frequency"]
        ds  = inputs["dropout_score"]
        ca  = inputs["completed_assignments"]
        sc  = (
            (1 - cr) * 30
            + min(la / 30, 1) * 25
            + ds * 25
            + (1 - min(lf / 7, 1)) * 10
            + (10 if ca < 3 else 0)
        )
        p2  = min(sc / 100, 0.95)
        p0r = max(0, 1 - p2) * (0.7 if cr > 0.65 else 0.4)
        p1r = max(0, 1 - p2 - p0r)
        tot = p0r + p1r + p2
        probs = [p0r / tot, p1r / tot, p2 / tot]
        pred  = int(np.argmax(probs))
    return dict(
        pred=pred, probs=probs,
        cr=inputs["completion_rate"],
        la=inputs["last_activity_days_ago"],
        lf=inputs["login_frequency"],
        ds=inputs["dropout_score"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — ประเมินนักเรียน
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔍  ประเมินนักเรียน":

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(
            '<div class="ph"><h2>ประเมินนักเรียน</h2>'
            '<p>กรอกข้อมูลให้ครบแล้วกดปุ่มประเมินด้านล่าง</p></div>',
            unsafe_allow_html=True,
        )

        # Mode badge
        mode_label = "โหมด ML จริง" if MODEL_READY else "โหมด Fallback (ไม่พบ models/pipeline.pkl)"
        mode_cls   = "mode-dot" if MODEL_READY else "mode-dot mode-dot-fb"
        st.markdown(
            f'<div class="mode-badge"><div class="{mode_cls}"></div>{mode_label}</div>',
            unsafe_allow_html=True,
        )

        # ── Card A: ข้อมูลพื้นฐาน ────────────────────────────────────────
        st.markdown('<div class="fcard"><div class="fcard-lbl">ข้อมูลพื้นฐาน</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2, gap="large")
        with a1:
            age    = st.slider("อายุ (ปี)", 15, 60, 22)
            region = st.selectbox("เมืองที่อยู่", [
                "Cairo", "Alexandria", "Amman", "Dubai", "Baghdad",
                "Doha", "Tunis", "Beirut", "Riyadh", "Casablanca",
            ])
        with a2:
            st.markdown('<div class="seg-wrap">', unsafe_allow_html=True)
            exam_season = st.radio(
                "ช่วงเวลา", [0, 1],
                format_func=lambda x: "ช่วงปกติ" if x == 0 else "ใกล้สอบ",
                horizontal=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            days_since_enroll = st.number_input(
                "สมัครมาแล้ว (วัน)", 1, 1825, 180,
                help="เช่น 6 เดือน = 180, 1 ปี = 365"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Card B: ความคืบหน้า ──────────────────────────────────────────
        st.markdown('<div class="fcard"><div class="fcard-lbl">ความคืบหน้าการเรียน</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3, gap="medium")
        with b1:
            courses_enrolled = st.number_input("วิชาที่ลงเรียน", 1, 20, 3)
        with b2:
            completed_assignments = st.number_input("งานที่ส่งแล้ว", 0, 200, 15)
        with b3:
            cr_pct = st.slider("ส่งงานครบ (%)", 0, 100, 60,
                               help="0% = ไม่ส่งเลย · 100% = ส่งครบทุกชิ้น")
        completion_rate = cr_pct / 100.0
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Card C: พฤติกรรม ─────────────────────────────────────────────
        st.markdown('<div class="fcard"><div class="fcard-lbl">พฤติกรรมการใช้งาน</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            login_frequency        = st.slider("เข้าระบบ / สัปดาห์ (ครั้ง)", 0.0, 30.0, 5.0, 0.5)
            last_activity_days_ago = st.number_input(
                "ไม่ได้เข้าระบบล่าสุด (วัน)", 0, 365, 7,
                help="เกิน 14 วัน = เริ่มน่าเป็นห่วง | เกิน 30 วัน = น่ากังวลมาก"
            )
        with c2:
            forum_posts_count = st.number_input("โพสต์ถามตอบ (ครั้ง)", 0, 500, 5)
            dropout_score     = st.slider(
                "คะแนนเสี่ยงจากระบบ", 0.0, 1.0, 0.30, 0.01,
                help="ค่าที่ระบบอื่นประเมิน: 0 = เสี่ยงน้อย · 1 = เสี่ยงมาก\n"
                     "< 0.35 ต่ำ | 0.35–0.65 ปานกลาง | > 0.65 สูง"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Validation ───────────────────────────────────────────────────
        warnings = []
        if completed_assignments > courses_enrolled * 50:
            warnings.append("จำนวนงานที่ส่ง (<b>{}</b>) สูงเกินสัดส่วนวิชาที่ลงเรียน ({} วิชา) — กรุณาตรวจสอบข้อมูล".format(
                completed_assignments, courses_enrolled))
        if completion_rate > 0.8 and last_activity_days_ago > 60:
            warnings.append("ส่งงานครบ <b>{}%</b> แต่ไม่ได้เข้าระบบมา <b>{} วัน</b> — ข้อมูลอาจขัดแย้งกัน".format(
                cr_pct, last_activity_days_ago))

        can_submit = True
        if warnings:
            for w in warnings:
                st.markdown(
                    f'<div class="val-banner"><div class="val-dot"></div>'
                    f'<div>⚠️ {w} (สามารถประเมินต่อได้ แต่ควรตรวจสอบก่อน)</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Submit ────────────────────────────────────────────────────────
        predict_clicked = st.button("ประเมินนักเรียน", type="primary")

    # ── Right panel ───────────────────────────────────────────────────────
    with right:
        st.markdown("<div style='height:56px'></div>", unsafe_allow_html=True)

        if "result" not in st.session_state:
            st.session_state.result = None
        if "loading" not in st.session_state:
            st.session_state.loading = False

        # Trigger prediction
        if predict_clicked:
            st.session_state.loading = True
            st.session_state.result  = None

            inputs = dict(
                age=age, region=region, exam_season=exam_season,
                courses_enrolled=courses_enrolled,
                completed_assignments=completed_assignments,
                completion_rate=completion_rate,
                login_frequency=login_frequency,
                last_activity_days_ago=last_activity_days_ago,
                forum_posts_count=forum_posts_count,
                dropout_score=dropout_score,
                days_since_enroll=days_since_enroll,
            )

            # Loading indicator
            with st.spinner(""):
                st.markdown("""
                <div class="loading-card">
                  <div class="spin"></div>
                  <div class="loading-txt">กำลังวิเคราะห์ข้อมูล…</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.6)   # brief pause so spinner registers

            result = run_prediction(inputs)
            st.session_state.result  = result
            st.session_state.loading = False
            st.rerun()

        if st.session_state.result is None:
            render_idle()
        else:
            render_result(st.session_state.result)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — เกี่ยวกับ ML
# ─────────────────────────────────────────────────────────────────────────────
else:
    hdr_col, back_col = st.columns([5, 1], gap="small")
    with hdr_col:
        st.markdown(
            '<div class="ph"><h2>เกี่ยวกับ ML</h2>'
            '<p>ระบบนี้ทำงานอย่างไร และข้อมูลใดบ้างที่ใช้ในการประเมิน</p></div>',
            unsafe_allow_html=True,
        )
    with back_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("← กลับ", key="back_btn"):
            st.rerun()

    # ── 1. ระบบคืออะไร ───────────────────────────────────────────────────
    st.markdown("""
    <div class="fcard">
      <div class="fcard-lbl">ระบบนี้คืออะไร</div>
      <p style="font-size:13px;color:#3a3a3c;line-height:1.7;margin:0">
        ระบบเฝ้าระวังการหยุดเรียน (Early Warning System) ใช้ Machine Learning
        วิเคราะห์พฤติกรรมการเรียนออนไลน์ เพื่อช่วยครูระบุนักเรียนที่ต้องการ
        ความช่วยเหลือก่อนที่จะสายเกินไป ผลการประเมินควรใช้ร่วมกับดุลยพินิจ
        ของครูเสมอ ไม่ใช่ตัดสินใจด้วยระบบเพียงอย่างเดียว
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Algorithm + Target ────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown("""
        <div class="fcard" style="height:100%">
          <div class="fcard-lbl">Algorithm ที่ใช้</div>
          <p style="font-size:13px;color:#3a3a3c;line-height:1.65;margin:0 0 12px">
            <strong style="color:#1c1c1e">Random Forest Classifier</strong> —
            รวม decision tree หลายต้นเพื่อลด overfitting train ด้วย 5-Fold CV
            และ tune hyperparameter ด้วย GridSearchCV
          </p>
          <span class="chip ch-b">Random Forest</span>
          <span class="chip ch-gr">GridSearchCV</span>
          <span class="chip ch-gr">5-Fold CV</span>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="fcard" style="height:100%">
          <div class="fcard-lbl">สิ่งที่ทำนาย (Target)</div>
          <p style="font-size:13px;color:#3a3a3c;line-height:1.65;margin:0 0 12px">
            จำแนกนักเรียนเป็น 3 สถานะ วัดด้วย
            <strong style="color:#1c1c1e">Macro F1-Score</strong>
            เพราะทั้ง 3 class มีจำนวนค่อนข้าง balanced
          </p>
          <span class="chip ch-g">เรียนปกติดี</span>
          <span class="chip ch-a">เริ่มมีสัญญาณเสี่ยง</span>
          <span class="chip ch-r">เสี่ยงหยุดเรียนสูง</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 3. Features ──────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2, gap="medium")
    with col_f1:
        st.markdown("""
        <div class="fcard">
          <div class="fcard-lbl">Features: พฤติกรรม</div>
          <span class="chip ch-b">completion_rate</span>
          <span class="chip ch-b">login_frequency</span>
          <span class="chip ch-b">last_activity_days_ago</span>
          <span class="chip ch-b">forum_posts_count</span>
          <span class="chip ch-b">dropout_score</span>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="fcard">
          <div class="fcard-lbl">Features: การเรียน & ทั่วไป</div>
          <span class="chip ch-g">courses_enrolled</span>
          <span class="chip ch-g">completed_assignments</span>
          <span class="chip ch-g">days_since_enroll</span>
          <span class="chip ch-gr">age</span>
          <span class="chip ch-gr">region</span>
          <span class="chip ch-gr">exam_season</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Feature importance ────────────────────────────────────────────
    st.markdown("""
    <div class="fcard">
      <div class="fcard-lbl">Features ที่สำคัญที่สุด</div>
      <p style="font-size:13px;color:#3a3a3c;line-height:1.7;margin:0">
        จากการวิเคราะห์ Feature Importance พบว่า
        <strong style="color:#1c1c1e">completion_rate</strong> และ
        <strong style="color:#1c1c1e">last_activity_days_ago</strong>
        มีผลต่อการทำนายมากที่สุด — สอดคล้องกับงานวิจัยที่พบว่านักเรียนที่
        หายไปนานและไม่ส่งงานมีแนวโน้ม dropout สูงกว่ามาก
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 5. Confidence guide ──────────────────────────────────────────────
    st.markdown("""
    <div class="fcard">
      <div class="fcard-lbl">วิธีตีความความมั่นใจ (%)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
        <div style="background:#eef7f2;border:.5px solid #9fd3b5;border-radius:10px;padding:12px">
          <div style="font-size:18px;font-weight:700;color:#1a6635;margin-bottom:4px">≥ 75%</div>
          <div style="font-size:12px;color:#1a6635;font-weight:500">ความมั่นใจสูง</div>
          <div style="font-size:11px;color:#3a6648;margin-top:4px;line-height:1.5">
            ผลน่าเชื่อถือ สามารถวางแผนการช่วยเหลือได้ทันที</div>
        </div>
        <div style="background:#fdf6e8;border:.5px solid #e0be7a;border-radius:10px;padding:12px">
          <div style="font-size:18px;font-weight:700;color:#7a4f00;margin-bottom:4px">55–74%</div>
          <div style="font-size:12px;color:#7a4f00;font-weight:500">ความมั่นใจปานกลาง</div>
          <div style="font-size:11px;color:#5a3a00;margin-top:4px;line-height:1.5">
            ควรรวบรวมข้อมูลเพิ่มก่อนตัดสินใจ</div>
        </div>
        <div style="background:#fdf0f0;border:.5px solid #e8a0a0;border-radius:10px;padding:12px">
          <div style="font-size:18px;font-weight:700;color:#7a1f1f;margin-bottom:4px">&lt; 55%</div>
          <div style="font-size:12px;color:#7a1f1f;font-weight:500">ความมั่นใจต่ำ</div>
          <div style="font-size:11px;color:#5a1010;margin-top:4px;line-height:1.5">
            พิจารณาร่วมกับข้อมูลอื่น ผลอาจคลาดเคลื่อน</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 6. Disclaimer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="note-box">
      <strong>ข้อควรระวัง —</strong>
      ผลที่ได้จากระบบนี้เป็นการประเมินเบื้องต้นเท่านั้น ไม่ใช่ผลสรุปขั้นสุดท้าย
      ควรใช้ร่วมกับการพิจารณาและดุลยพินิจของครูผู้สอนเสมอ
      ระบบมีความถูกต้องสูงสุดเมื่อข้อมูลที่กรอกครบถ้วนและเป็นปัจจุบัน
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("← กลับไปหน้าประเมินนักเรียน", key="back_btn2"):
        st.rerun()