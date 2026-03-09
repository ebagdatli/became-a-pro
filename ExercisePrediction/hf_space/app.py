"""
BecomeAPro - AI-Powered Exercise Tracker (Redesigned UI)
Streamlit + WebRTC for in-browser real-time pose detection.
"""
import json
import logging
import os
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path
from threading import Lock

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

try:
    from streamlit_webrtc import get_twilio_ice_servers
except ImportError:
    get_twilio_ice_servers = None

# Heavy libraries are imported lazily inside functions to minimize startup memory
# av, cv2, mediapipe, joblib are NOT imported at module level

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
if not (MODELS_DIR / "meta.pkl").exists():
    MODELS_DIR = ROOT
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

BUFFER_SIZE = 12
CONFIDENCE_THRESHOLD = 0.65
SCALE_XY = 100.0
SCALE_Z = 200.0
FRAME_SKIP = 2
REP_DEBOUNCE = 3
REP_DISPLAY_FRAMES = 20

KCAL_PER_REP = {
    "pushups": 0.4,
    "situp": 0.3,
    "squats": 0.35,
    "pullups": 0.5,
    "jumping_jacks": 0.2,
}

BODY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

MP_INDEX_TO_NAME = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NAME_ALIASES = {
    "right_index_1": "right_index", "left_index_1": "left_index",
    "left_pinky_1": "left_pinky", "right_pinky_1": "right_pinky",
}

POSE_TO_TURKISH = {
    "situp_up": "Mekik (Yukari)",
    "situp_down": "Mekik (Asagi)",
    "pushups_up": "Sinav (Yukari)",
    "pushups_down": "Sinav (Asagi)",
    "pullups_up": "Barfiks (Yukari)",
    "pullups_down": "Barfiks (Asagi)",
    "squats_up": "Squat (Yukari)",
    "squats_down": "Squat (Asagi)",
    "jumping_jacks_up": "Ziplama (Yukari)",
    "jumping_jacks_down": "Ziplama (Asagi)",
    "Belirsiz": "Belirsiz",
}

EXERCISES = [
    {"name": "Sinav",   "en": "Push-ups",      "icon": "💪", "code": "PUSH",  "desc": "Göğüs, omuz ve triceps kasları için temel egzersiz."},
    {"name": "Mekik",   "en": "Sit-ups",       "icon": "🔄", "code": "SIT",   "desc": "Karın kasları için etkili bir core egzersizi."},
    {"name": "Squat",   "en": "Squats",        "icon": "🦵", "code": "SQUAT", "desc": "Bacak ve kalça kasları için en etkili hareket."},
    {"name": "Barfiks", "en": "Pull-ups",      "icon": "🧗", "code": "PULL",  "desc": "Sırt ve biceps kaslarını güçlendiren egzersiz."},
    {"name": "Ziplama", "en": "Jumping Jacks", "icon": "🤸", "code": "JUMP",  "desc": "Tam vücut kardiyo ve koordinasyon egzersizi."},
]

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BecomeAPro | AI Exercise Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """\
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:         #0b0f0e;
  --bg2:        #111916;
  --bg3:        #162018;
  --surface:    rgba(255,255,255,0.032);
  --border:     rgba(255,255,255,0.07);
  --border-acc: rgba(180,255,60,0.22);
  --lime:       #b4ff3c;
  --lime-dim:   #7ab828;
  --lime-glow:  rgba(180,255,60,0.12);
  --amber:      #ffb830;
  --muted:      #4a5550;
  --sub:        #7a9a8e;
  --text:       #dde8e3;
  --white:      #ffffff;
  --r:          14px;
  --rl:         22px;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    max-width: 1060px;
    margin: 0 auto;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHorizontalBlock"] { gap: 1rem !important; align-items: stretch !important; }
[data-testid="stColumn"] { display: flex !important; flex-direction: column !important; }
[data-testid="stColumn"] > div { flex: 1; }

/* ── HERO ── */
.hero {
    padding: 4.5rem 0 3rem;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 180px; height: 2px;
    background: var(--lime);
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--lime);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0.9;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 18px; height: 2px;
    background: var(--lime);
    flex-shrink: 0;
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.2rem, 7.5vw, 5.8rem);
    line-height: 0.95;
    letter-spacing: 1.5px;
    color: var(--white);
    margin: 0 0 1.2rem;
    font-weight: 400;
}
.hero h1 em { font-style: normal; color: var(--lime); }
.hero-sub {
    font-size: 0.97rem;
    color: var(--sub);
    line-height: 1.75;
    font-weight: 300;
    max-width: 400px;
}
.hero-meta {
    text-align: right;
    padding-bottom: 0.4rem;
}
.hero-version {
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.hero-tags {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    line-height: 2.4;
    letter-spacing: 0.5px;
}

/* ── STATS ── */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-bottom: 1px solid var(--border);
}
.stat-item {
    padding: 2rem 1.5rem;
    position: relative;
}
.stat-item:not(:last-child)::after {
    content: '';
    position: absolute;
    right: 0; top: 22%; bottom: 22%;
    width: 1px;
    background: var(--border);
}
.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    line-height: 1;
    color: var(--white);
    letter-spacing: 1px;
}
.stat-num sup {
    font-size: 1.4rem;
    color: var(--lime);
    vertical-align: super;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── SECTION HEADER ── */
.sec-hdr {
    padding: 3rem 0 1.8rem;
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sec-idx {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--lime-dim);
    letter-spacing: 2px;
    flex-shrink: 0;
}
.sec-ttl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.1rem;
    letter-spacing: 1.5px;
    color: var(--white);
    font-weight: 400;
    line-height: 1;
}
.sec-note {
    font-size: 0.8rem;
    color: var(--muted);
    margin-left: auto;
    flex-shrink: 0;
    font-weight: 400;
}

/* ── STEP CARDS ── */
.step-card {
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.8rem 1.5rem 1.5rem;
    background: var(--surface);
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, background 0.25s;
}
.step-card:hover {
    border-color: var(--border-acc);
    background: var(--bg3);
}
.step-card:hover .step-bg-n { color: rgba(180,255,60,0.08); }
.step-bg-n {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    line-height: 1;
    color: rgba(255,255,255,0.03);
    position: absolute;
    bottom: -0.5rem; right: 1rem;
    letter-spacing: 1px;
    pointer-events: none;
    transition: color 0.3s;
    user-select: none;
}
.step-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--lime-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    display: block;
}
.step-ico { font-size: 1.5rem; margin-bottom: 0.8rem; display: block; }
.step-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 0.45rem;
    line-height: 1.3;
}
.step-desc { font-size: 0.82rem; color: var(--sub); line-height: 1.65; font-weight: 300; }

/* ── EXERCISE CARDS ── */
.ex-card {
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem 1.1rem 1.3rem;
    background: var(--surface);
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.ex-card:hover {
    border-color: var(--border-acc);
    transform: translateY(-4px);
    background: var(--bg3);
    box-shadow: 0 20px 50px rgba(0,0,0,0.3), 0 0 0 1px rgba(180,255,60,0.1);
}
.ex-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--lime-dim);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: block;
    opacity: 0.8;
}
.ex-ico { font-size: 1.8rem; margin-bottom: 0.9rem; display: block; }
.ex-tr {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: var(--white);
    letter-spacing: 1px;
    line-height: 1;
    margin-bottom: 0.15rem;
    font-weight: 400;
}
.ex-en { font-size: 0.7rem; color: var(--muted); margin-bottom: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.ex-info { font-size: 0.78rem; color: var(--sub); line-height: 1.5; font-weight: 300; }

/* ── CTA SECTION ── */
.cta-section {
    margin-top: 3rem;
    border: 1px solid var(--border);
    border-radius: var(--rl);
    overflow: hidden;
    background: var(--bg2);
}
.cta-top {
    padding: 2.8rem 3rem 2.4rem;
    background: radial-gradient(ellipse 55% 90% at 5% 10%, rgba(180,255,60,0.05) 0%, transparent 65%);
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 3rem;
}
.cta-text { }
.cta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: var(--lime);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cta-tag::before { content: ''; width: 14px; height: 2px; background: var(--lime); display: inline-block; flex-shrink: 0; }
.cta-ttl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 1.5px;
    color: var(--white);
    font-weight: 400;
    margin-bottom: 0.6rem;
    line-height: 1;
}
.cta-sub {
    font-size: 0.87rem;
    color: var(--sub);
    line-height: 1.7;
    max-width: 400px;
    font-weight: 300;
    margin-bottom: 1.6rem;
}
/* Fake start button shown in CTA header (decorative, real one is from webrtc below) */
.cta-start-btn {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: var(--lime);
    color: #0b0f0e;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.8rem 1.8rem;
    border-radius: 10px;
    cursor: default;
    pointer-events: none;
    opacity: 0.95;
}
.cta-start-btn .btn-ico {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(0,0,0,0.15);
    font-size: 0.75rem;
}
.cta-start-btn .btn-hint {
    font-size: 0.7rem;
    font-weight: 400;
    opacity: 0.6;
    margin-left: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0;
}
/* checklist items */
.cta-checks {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.8rem;
}
.cta-check {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: var(--sub);
    font-weight: 300;
}
.cta-check::before {
    content: '✓';
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px; height: 18px;
    border-radius: 5px;
    background: rgba(180,255,60,0.1);
    border: 1px solid rgba(180,255,60,0.2);
    color: var(--lime);
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
}
/* right side visual */
.cta-visual {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}
.cta-cam-icon {
    width: 110px; height: 110px;
    border: 1px solid var(--border-acc);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(180,255,60,0.03);
    position: relative;
    font-size: 2.8rem;
}
.cta-cam-icon::before {
    content: '';
    position: absolute;
    inset: -10px;
    border-radius: 50%;
    border: 1px dashed rgba(180,255,60,0.12);
    animation: spin 12s linear infinite;
}
.cta-cam-icon::after {
    content: '';
    position: absolute;
    inset: -20px;
    border-radius: 50%;
    border: 1px dashed rgba(180,255,60,0.06);
    animation: spin 18s linear infinite reverse;
}
@keyframes spin { to { transform: rotate(360deg); } }
.cta-cam-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--lime-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.cta-body { padding: 2rem 3rem 2.5rem; }

/* ── TROUBLE TOGGLE ── */
.trouble-toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 1px;
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px;
    background: transparent;
    transition: border-color 0.2s, color 0.2s;
    margin-top: 1rem;
    width: fit-content;
}
.trouble-toggle:hover { border-color: rgba(255,184,48,0.2); color: #b0a080; }
.trouble-panel {
    display: none;
    margin-top: 0.8rem;
    background: rgba(255,184,48,0.03);
    border: 1px solid rgba(255,184,48,0.1);
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    font-size: 0.79rem;
    color: #a09060;
    line-height: 1.65;
}
.trouble-panel strong { color: #c4a468; }

/* ── CAM ── */
.cam-wrap {
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
    background: #070d0a;
}
.cam-bar {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,0,0,0.25);
}
.cam-pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--lime);
    box-shadow: 0 0 6px var(--lime);
    animation: pulse 1.8s ease-in-out infinite;
}
.cam-pulse.off { background: var(--muted); box-shadow: none; animation: none; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.35;transform:scale(0.75)} }
.cam-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.cam-live {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--lime);
    letter-spacing: 2px;
    text-transform: uppercase;
}



/* ── STATUS PILL ── */
.status-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 0.6rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--lime-dim);
    letter-spacing: 0.5px;
}
.s-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--lime); animation: pulse 1.5s ease-in-out infinite; }

/* ── SUMMARY ── */
.summary {
    border: 1px solid var(--border-acc);
    border-radius: var(--rl);
    overflow: hidden;
    max-width: 500px;
    margin: 2rem auto;
    background: var(--bg2);
}
.sum-head {
    padding: 1.4rem 1.8rem;
    background: rgba(180,255,60,0.04);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
}
.sum-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 1.5px;
    color: var(--white);
    font-weight: 400;
}
.sum-dur {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 1px;
}
.sum-body { padding: 0 1.8rem; }
.sum-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
}
.sum-row:last-child { border-bottom: none; }
.sum-ex { font-size: 0.87rem; color: var(--text); font-weight: 500; }
.sum-rep {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: var(--lime);
    font-weight: 700;
}
.sum-foot {
    padding: 1.1rem 1.8rem;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sum-kcal-lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.sum-kcal-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--amber);
    letter-spacing: 1px;
}
.sum-kcal-unit { font-size: 0.72rem; color: var(--muted); margin-left: 4px; }

/* ── BUTTONS ── */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--lime) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2.4rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #0b0f0e !important;
    letter-spacing: 0.2px !important;
    min-height: 50px !important;
    transition: all 0.2s !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 0 30px rgba(180,255,60,0.28) !important;
    transform: translateY(-2px) !important;
    background: #c4ff52 !important;
}

/* ── NO MODEL ── */
.no-model {
    border: 1px dashed rgba(180,255,60,0.12);
    border-radius: var(--rl);
    padding: 3.5rem 2rem;
    text-align: center;
    background: var(--surface);
}
.no-model-ico { font-size: 2.4rem; margin-bottom: 1rem; display: block; opacity: 0.6; }
.no-model h3 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    color: var(--white);
    font-weight: 400;
    margin-bottom: 0.6rem;
    letter-spacing: 1.5px;
}
.no-model p { font-size: 0.84rem; color: var(--sub); line-height: 1.7; max-width: 420px; margin: 0 auto 0.5rem; }
.no-model code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    background: rgba(180,255,60,0.06);
    color: var(--lime-dim);
    padding: 2px 8px;
    border-radius: 5px;
    border: 1px solid rgba(180,255,60,0.12);
}

/* ── FOOTER ── */
.foot {
    padding: 2rem 0 1.8rem;
    border-top: 1px solid var(--border);
    margin-top: 4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.foot-brand {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    letter-spacing: 3px;
}
.foot-stack {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 1px;
    opacity: 0.7;
}

::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1c2820; border-radius: 3px; }
</style>
"""

# ── ICE / TURN ─────────────────────────────────────────────────────────────
def get_ice_config() -> dict:
    if get_twilio_ice_servers is not None:
        try:
            sid   = os.environ.get("TWILIO_ACCOUNT_SID", "") or st.secrets.get("TWILIO_ACCOUNT_SID", "")
            token = os.environ.get("TWILIO_AUTH_TOKEN", "") or st.secrets.get("TWILIO_AUTH_TOKEN", "")
            if sid and token:
                ice = get_twilio_ice_servers(twilio_sid=sid, twilio_token=token)
                return {"iceServers": ice}
        except Exception as exc:
            logger.warning("Twilio ICE fetch failed: %s", exc)
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ── Pose helpers ────────────────────────────────────────────────────────────
def label_to_turkish(label: str) -> str:
    return POSE_TO_TURKISH.get(label, label)

def ensure_pose_model() -> str:
    """Download pose model if not present. Uses spinner on first download."""
    if POSE_MODEL_PATH.exists():
        return str(POSE_MODEL_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = POSE_MODEL_PATH.with_suffix(".task.tmp")
    try:
        with st.spinner("Pose modeli indiriliyor (ilk acilis)..."):
            urllib.request.urlretrieve(POSE_MODEL_URL, tmp_path)
        tmp_path.rename(POSE_MODEL_PATH)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Pose model indirilemedi: {exc}") from exc
    return str(POSE_MODEL_PATH)

def landmarks_to_vector(landmark_list, feature_columns):
    name_to_idx = {name: i for i, name in enumerate(MP_INDEX_TO_NAME)}
    for alias, canonical in NAME_ALIASES.items():
        name_to_idx[alias] = name_to_idx.get(canonical, 0)
    values = []
    for col in feature_columns:
        if not col.startswith(("x_", "y_", "z_")):
            continue
        axis = col[0]; name = NAME_ALIASES.get(col[2:].strip(), col[2:].strip())
        idx = name_to_idx.get(name, -1)
        if idx < 0: values.append(0.0); continue
        lm = landmark_list[idx]
        x_val = lm.x if lm.x is not None else 0.0
        y_val = lm.y if lm.y is not None else 0.0
        z_val = lm.z if lm.z is not None else 0.0
        if   axis == "x": values.append((x_val - 0.5) * SCALE_XY)
        elif axis == "y": values.append((y_val - 0.5) * SCALE_XY)
        else:             values.append(z_val * SCALE_Z)
    return np.array(values, dtype=np.float32).reshape(1, -1)

def predict_single(ml_model, encoder, scaler, model_type, X, buffer):
    X_scaled = scaler.transform(X)
    if model_type == "xgboost":
        pred_idx = ml_model.predict(X_scaled)[0]
        probs    = ml_model.predict_proba(X_scaled)[0]
    else:
        import torch
        with torch.no_grad():
            X_t    = torch.from_numpy(X_scaled.astype(np.float32))
            logits = ml_model(X_t)
            probs  = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    buffer.append("Belirsiz" if conf < CONFIDENCE_THRESHOLD else encoder.inverse_transform([pred_idx])[0])
    mode_label = Counter(buffer).most_common(1)[0][0]
    return mode_label, conf

def draw_overlay_panel(frame, label, conf, reps=None):
    import cv2 as _cv2
    h, w = frame.shape[:2]
    has_reps = reps is not None and reps > 0
    panel_h = 120 if has_reps else 90
    panel_w = min(400, w - 20)
    x1, y1, x2, y2 = 10, 10, 10 + panel_w, 10 + panel_h
    overlay = frame.copy()
    _cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 22, 18), -1)
    _cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    _cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 220, 40), 2)
    turkce = label_to_turkish(label)
    font = _cv2.FONT_HERSHEY_SIMPLEX
    color = (90, 255, 60) if label != "Belirsiz" else (80, 80, 80)
    _cv2.putText(frame, f"Hareket: {turkce}", (x1+12, y1+38), font, 0.9, color, 2)
    _cv2.putText(frame, f"Guven: %{conf*100:.0f}", (x1+12, y1+72), font, 0.7, (180,200,180), 2)
    if has_reps:
        _cv2.putText(frame, f"Tekrar: {reps}", (x1+12, y1+106), font, 0.8, (90,220,40), 2)

def draw_center_counter(frame, reps, frames_since_rep):
    import cv2 as _cv2
    if frames_since_rep >= REP_DISPLAY_FRAMES: return
    alpha = 1.0 - (frames_since_rep / REP_DISPLAY_FRAMES)
    h, w = frame.shape[:2]
    text = str(reps)
    font = _cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 4.0, 8
    (tw, th), _ = _cv2.getTextSize(text, font, scale, thickness)
    overlay = frame.copy()
    _cv2.putText(overlay, text, ((w-tw)//2, (h+th)//2), font, scale, (180,255,60), thickness)
    _cv2.addWeighted(overlay, alpha*0.65, frame, 1.0-alpha*0.65, 0, frame)

# ── Artifacts ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_artifacts():
    """Load only the ML model files (no network calls). Returns None tuple if missing."""
    from joblib import load as jload
    meta_path     = MODELS_DIR / "meta.pkl"
    metadata_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists() or not metadata_path.exists():
        return None, None, None, None, None, None
    try:
        meta       = jload(meta_path)
        encoder    = jload(MODELS_DIR / "encoder.pkl")
        scaler     = jload(MODELS_DIR / "scaler.pkl")
        model_type = meta.get("model_type", "xgboost")
        model_path = meta.get("model_path")
        if model_path:
            filename   = model_path.replace("\\", "/").split("/")[-1]
            model_path = MODELS_DIR / filename
        if model_type == "xgboost":
            ml_model = jload(model_path)
        else:
            import torch
            from torch import nn
            input_size  = meta.get("input_size", 99)
            num_classes = meta.get("num_classes", 10)
            ml_model = nn.Sequential(
                nn.Linear(input_size, 200), nn.ReLU(), nn.Linear(200, num_classes)
            )
            ml_model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            ml_model.eval()
        with open(metadata_path, encoding="utf-8") as f:
            feature_columns = json.load(f).get("feature_columns", [])
        return ml_model, encoder, scaler, model_type, feature_columns, meta
    except Exception as exc:
        logger.error("ML artifact load failed: %s", exc)
        return None, None, None, None, None, None


@st.cache_resource
def load_pose_landmarker():
    """Load MediaPipe pose landmarker (downloads model on first run). Lazy import."""
    try:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _vision
        pose_model_path = ensure_pose_model()
        base_options    = _mp_python.BaseOptions(model_asset_path=pose_model_path)
        options         = _vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=_vision.RunningMode.IMAGE,
        )
        return _vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        logger.error("Pose landmarker load failed: %s", exc)
        return None


def load_all_artifacts():
    """Compatibility wrapper — loads ML artifacts then pose landmarker separately."""
    ml_model, encoder, scaler, model_type, feature_columns, meta = load_ml_artifacts()
    if ml_model is None:
        return None, None, None, None, None, None, None
    pose_landmarker = load_pose_landmarker()
    if pose_landmarker is None:
        return None, None, None, None, None, None, None
    return ml_model, encoder, scaler, model_type, feature_columns, pose_landmarker, meta

# ── WebRTC callback ─────────────────────────────────────────────────────────
_buffer_lock = Lock()
_prediction_buffer: deque = deque(maxlen=BUFFER_SIZE)

def _draw_body_skeleton(img, pose_landmarks):
    import cv2 as _cv2
    h, w = img.shape[:2]
    points = {}
    for idx in BODY_LANDMARK_INDICES:
        lm = pose_landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        points[idx] = (px, py)
        _cv2.circle(img, (px, py), 5, (90, 220, 40), -1)
        _cv2.circle(img, (px, py), 7, (90, 220, 40), 1)
    for a, b in BODY_CONNECTIONS:
        if a in points and b in points:
            _cv2.line(img, points[a], points[b], (60, 200, 20), 2)

def make_video_frame_callback(ml_model, encoder, scaler, model_type, feature_columns, pose_landmarker):
    frame_counter = [0]
    cached_label  = ["Belirsiz"]
    cached_conf   = [0.0]
    rep_state = {
        "phase": "idle", "reps": 0, "debounce_count": 0,
        "pending_phase": None, "frames_since_rep": REP_DISPLAY_FRAMES,
        "exercise_reps": {}, "start_time": None,
    }

    def _update_rep_counter(label):
        phase = rep_state["phase"]
        if rep_state["start_time"] is None and label != "Belirsiz":
            rep_state["start_time"] = time.time()
        exercise = label.rsplit("_", 1)[0] if "_" in label else None
        if   label.endswith("_down"): target = "down"
        elif label.endswith("_up"):   target = "up"
        else:
            rep_state["debounce_count"] = 0; rep_state["pending_phase"] = None; return
        if   phase == "idle" and target == "down":  _try_transition("down", exercise)
        elif phase == "down" and target == "up":
            if _try_transition("up", exercise):
                rep_state["reps"] += 1; rep_state["frames_since_rep"] = 0
                if exercise:
                    rep_state["exercise_reps"][exercise] = rep_state["exercise_reps"].get(exercise, 0) + 1
        elif phase == "up"   and target == "down":  _try_transition("down", exercise)

    def _try_transition(target, exercise=None):
        if rep_state["pending_phase"] == target: rep_state["debounce_count"] += 1
        else: rep_state["pending_phase"] = target; rep_state["debounce_count"] = 1
        if rep_state["debounce_count"] >= REP_DEBOUNCE:
            rep_state["phase"] = target; rep_state["pending_phase"] = None; rep_state["debounce_count"] = 0; return True
        return False

    def video_frame_callback(frame):
        import av as _av
        import cv2 as _cv2
        import mediapipe as _mp
        img = frame.to_ndarray(format="bgr24")
        img = _cv2.flip(img, 1)
        frame_counter[0] += 1; rep_state["frames_since_rep"] += 1
        if frame_counter[0] % FRAME_SKIP != 0:
            draw_overlay_panel(img, cached_label[0], cached_conf[0], reps=rep_state["reps"])
            draw_center_counter(img, rep_state["reps"], rep_state["frames_since_rep"])
            return _av.VideoFrame.from_ndarray(img, format="bgr24")
        rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        try:
            detection_result = pose_landmarker.detect(mp_image)
        except Exception:
            draw_overlay_panel(img, "Belirsiz", 0.0, reps=rep_state["reps"])
            return _av.VideoFrame.from_ndarray(img, format="bgr24")
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            _draw_body_skeleton(img, pose_landmarks)
            try:
                X = landmarks_to_vector(pose_landmarks, feature_columns)
                if X.shape[1] == scaler.n_features_in_:
                    with _buffer_lock:
                        label, conf = predict_single(ml_model, encoder, scaler, model_type, X, _prediction_buffer)
                    cached_label[0] = label; cached_conf[0] = conf
                    _update_rep_counter(label)
                    draw_overlay_panel(img, label, conf, reps=rep_state["reps"])
            except Exception as e:
                _cv2.putText(img, f"Err: {e}"[:60], (10,30), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        else:
            cached_label[0] = "Belirsiz"; cached_conf[0] = 0.0
            draw_overlay_panel(img, "Belirsiz", 0.0, reps=rep_state["reps"])
            h, w = img.shape[:2]
            _cv2.putText(img, "Tam vucut gorunumunde durun", (10, h-25), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,160,255), 1)
        draw_center_counter(img, rep_state["reps"], rep_state["frames_since_rep"])
        return _av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback, rep_state

# ── UI Sections ─────────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
        <div class="hero">
            <div>
                <div class="hero-eyebrow">AI-Powered Fitness Tracker</div>
                <h1>EGZERSIZINI<br><em>YAPAY ZEKA</em><br>ILE TAKIP ET</h1>
                <p class="hero-sub">
                    Kameranı aç, egzersizini yap.&nbsp;
                    Yapay zeka hareketlerini anlık olarak tanır,
                    tekrarlarını sayar ve performansını takip eder.
                </p>
            </div>
            <div class="hero-meta">
                <div class="hero-version">v2.0</div>
                <div class="hero-tags">
                    MediaPipe<br>XGBoost / PyTorch<br>Streamlit WebRTC
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_stats():
    st.markdown("""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-num">5<sup>✦</sup></div>
                <div class="stat-label">Desteklenen Egzersiz</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">10</div>
                <div class="stat-label">Hareket Pozisyonu</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">33</div>
                <div class="stat-label">Vücut Noktası Takibi</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_how_it_works():
    st.markdown("""
        <div class="sec-hdr">
            <span class="sec-idx">01 —</span>
            <span class="sec-ttl">NASIL ÇALIŞIR</span>
            <span class="sec-note">3 adımda antrenman</span>
        </div>
    """, unsafe_allow_html=True)

    steps = [
        ("📷", "ADIM 01", "Kamerayı Başlat",
         "START butonuna tıklayarak tarayıcı kameranızı açın. "
         "Kameranın tam vücudunuzu göreceği bir konumda durun."),
        ("🏋️", "ADIM 02", "Egzersizini Yap",
         "Şınav, mekik, squat veya başka bir egzersiz yapmaya başlayın. "
         "AI modeli hareketlerinizi anlık olarak tanır."),
        ("📊", "ADIM 03", "Sonuçlarını Gör",
         "Hareket tipi, güven oranı ve tekrar sayısı video üzerinde "
         "canlı olarak gösterilir. Durduğunda özet ekrana gelir."),
    ]
    cols = st.columns(3)
    for col, (ico, tag, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
                <div class="step-card">
                    <span class="step-bg-n">{tag[-2:]}</span>
                    <span class="step-tag">{tag}</span>
                    <span class="step-ico">{ico}</span>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)


def render_exercises():
    st.markdown("""
        <div class="sec-hdr">
            <span class="sec-idx">02 —</span>
            <span class="sec-ttl">DESTEKlENEN EGZERSİZLER</span>
            <span class="sec-note">AI destekli tanıma</span>
        </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for col, ex in zip(cols, EXERCISES):
        with col:
            st.markdown(f"""
                <div class="ex-card">
                    <span class="ex-code">{ex['code']}</span>
                    <span class="ex-ico">{ex['icon']}</span>
                    <div class="ex-tr">{ex['name']}</div>
                    <div class="ex-en">{ex['en']}</div>
                    <div class="ex-info">{ex['desc']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_camera_section(ml_model, encoder, scaler, model_type, feature_columns, pose_landmarker):
    # ── CTA header ──
    st.markdown("""
        <div class="cta-section">
            <div class="cta-top">
                <div class="cta-text">
                    <div class="cta-tag">03 — Antrenman Modu</div>
                    <div class="cta-ttl">ANTRENMANINA<br>BAŞLA</div>
                    <div class="cta-sub">
                        Kameranı açmak için aşağıdaki
                        <strong style="color:var(--lime)">KAMERAYI BAŞLAT</strong> butonuna tıkla,
                        izni onayla ve egzersizine başla.
                    </div>
                    <div class="cta-checks">
                        <span class="cta-check">İyi aydınlatılmış bir ortamda dur</span>
                        <span class="cta-check">Tam vücut kameraya görünsün</span>
                        <span class="cta-check">Chrome veya Edge önerilir</span>
                    </div>
                </div>
                <div class="cta-visual">
                    <div class="cta-cam-icon">📷</div>
                    <span class="cta-cam-label">Canlı Takip</span>
                </div>
            </div>
            <div class="cta-body">
    """, unsafe_allow_html=True)

    callback, rep_state = make_video_frame_callback(
        ml_model, encoder, scaler, model_type, feature_columns, pose_landmarker,
    )

    _pad_l, cam_col, _pad_r = st.columns([1, 6, 1])
    with cam_col:
        st.markdown("""
            <div class="cam-wrap">
                <div class="cam-bar">
                    <div class="cam-pulse off"></div>
                    <span class="cam-lbl">Kamera</span>
                </div>
        """, unsafe_allow_html=True)

        webrtc_ctx = webrtc_streamer(
            key="exercise-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=callback,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
            rtc_configuration=get_ice_config(),
            translations={"start": "KAMERAYI BAŞLAT", "stop": "DURDUR", "select_device": "Kamera Seç"},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if webrtc_ctx.state.playing:
        with cam_col:
            st.markdown("""
                <div class="status-row">
                    <span class="s-dot"></span>
                    KAMERA AKTİF — EGZERSİZE BAŞLAYIN
                </div>
            """, unsafe_allow_html=True)
        if rep_state["start_time"] is None:
            rep_state["start_time"] = time.time()
        st.session_state["rep_state_snapshot"] = {
            "reps": rep_state["reps"],
            "exercise_reps": dict(rep_state["exercise_reps"]),
            "start_time": rep_state["start_time"],
        }
    else:
        # Troubleshoot toggle (collapsible via JS)
        with cam_col:
            st.markdown("""
                <button class="trouble-toggle"
                    onclick="
                        var p=this.nextElementSibling;
                        p.style.display=p.style.display==='block'?'none':'block';
                        this.style.borderColor=p.style.display==='block'?'rgba(255,184,48,0.25)':'var(--border)';
                        this.style.color=p.style.display==='block'?'#c4a468':'var(--muted)';
                    ">
                    ⚠ Sorun mu yaşıyorsun?
                </button>
                <div class="trouble-panel">
                    <strong>Bağlantı sorunu mu?</strong>
                    Tarayıcınızın kamera erişimine izin verdiğinden emin olun.
                    Sorun devam ederse sayfayı yenileyip tekrar deneyin.
                </div>
            """, unsafe_allow_html=True)

        if st.session_state.get("rep_state_snapshot"):
            snap = st.session_state["rep_state_snapshot"]
            if snap["reps"] > 0 and snap["start_time"]:
                elapsed = time.time() - snap["start_time"]
                _render_workout_summary(snap, elapsed)
                st.session_state["rep_state_snapshot"] = None

    st.markdown("</div></div>", unsafe_allow_html=True)  # close cta-body + cta-section


def _render_workout_summary(snap, elapsed_seconds):
    mins = int(elapsed_seconds) // 60
    secs = int(elapsed_seconds) % 60
    total_kcal = 0.0
    exercise_names = {
        "pushups": "Şınav", "situp": "Mekik",
        "squats": "Squat", "pullups": "Barfiks", "jumping_jacks": "Zıplama",
    }
    rows_html = ""
    for ex, count in snap["exercise_reps"].items():
        name  = exercise_names.get(ex, ex)
        kcal  = count * KCAL_PER_REP.get(ex, 0.3)
        total_kcal += kcal
        rows_html += f"""
        <div class="sum-row">
            <span class="sum-ex">{name}</span>
            <span class="sum-rep">{count} tekrar</span>
        </div>"""
    st.markdown(f"""
        <div class="summary">
            <div class="sum-head">
                <span class="sum-title">ANTRENMAN ÖZETİ</span>
                <span class="sum-dur">{mins:02d}:{secs:02d}</span>
            </div>
            <div class="sum-body">{rows_html}</div>
            <div class="sum-foot">
                <span class="sum-kcal-lbl">Tahmini Kalori</span>
                <div>
                    <span class="sum-kcal-val">{total_kcal:.1f}</span>
                    <span class="sum-kcal-unit">kcal</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
        <div class="foot">
            <span class="foot-brand">BECOMEAPRO</span>
            <span class="foot-stack">MediaPipe &nbsp;·&nbsp; XGBoost / PyTorch &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; WebRTC</span>
        </div>
    """, unsafe_allow_html=True)


def render_model_missing():
    _p1, col_c, _p2 = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
            <div style="margin-top: 3rem;">
                <div class="no-model">
                    <span class="no-model-ico">📂</span>
                    <h3>MODEL DOSYALARI BULUNAMADI</h3>
                    <p>
                        Uygulamanın çalışabilmesi için eğitilmiş model dosyalarının
                        <code>models/</code> klasörüne eklenmesi gerekiyor.
                    </p>
                    <p style="margin-top:0.8rem;">
                        Gerekli dosyalar: <code>meta.pkl</code> · <code>encoder.pkl</code> ·
                        <code>scaler.pkl</code> · <code>final_model.pkl</code> · <code>metadata.json</code>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Step 1: Load ML model files only (fast, no network I/O) — page renders immediately
    ml_model, encoder, scaler, model_type, feature_columns, meta = load_ml_artifacts()

    render_hero()

    if ml_model is None:
        render_model_missing()
        render_footer()
        return

    render_stats()
    render_how_it_works()
    render_exercises()

    # Step 2: Load pose landmarker (may download ~8 MB on first run — spinner shown inside)
    pose_landmarker = load_pose_landmarker()
    if pose_landmarker is None:
        st.error("Pose modeli yuklenemedi. Lutfen sayfayi yenileyip tekrar deneyin.")
        render_footer()
        return

    render_camera_section(ml_model, encoder, scaler, model_type, feature_columns, pose_landmarker)
    render_footer()


if __name__ == "__main__":
    main()