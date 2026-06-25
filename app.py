import streamlit as st
from transformers import pipeline
import re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verifai — Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp { background-color: #0a0c14; color: #e8eaf6; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }

    .app-header { text-align: center; padding: 2rem 0 1.5rem; border-bottom: 1px solid #1e2540; margin-bottom: 2rem; }
    .app-header h1 { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #8b85ff; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
    .app-header p { color: #6b7399; font-size: 0.85rem; font-family: 'Space Mono', monospace; letter-spacing: 0.08em; }

    .section-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #6b7399; margin-bottom: 0.5rem; }

    .verdict-card { border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
    .verdict-real     { background: linear-gradient(135deg, #052e16, #0a3d1f); border: 1px solid rgba(34,197,94,0.4); border-left: 4px solid #22c55e; }
    .verdict-fake     { background: linear-gradient(135deg, #2d0a0a, #3d0f0f); border: 1px solid rgba(239,68,68,0.4);  border-left: 4px solid #ef4444; }
    .verdict-uncertain{ background: linear-gradient(135deg, #2d1a00, #3d2300); border: 1px solid rgba(245,158,11,0.4); border-left: 4px solid #f59e0b; }

    .verdict-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.3rem; }
    .verdict-value-real      { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #22c55e; }
    .verdict-value-fake      { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #ef4444; }
    .verdict-value-uncertain { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #f59e0b; }

    .signals-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.5rem 0 1rem; }
    .badge { font-family: 'Space Mono', monospace; font-size: 0.68rem; padding: 4px 10px; border-radius: 6px; display: inline-block; }
    .badge-high   { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-medium { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .badge-low    { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }

    .check-row   { display: flex; align-items: center; gap: 10px; margin: 0.4rem 0; font-size: 0.88rem; color: #b0b8d8; }
    .check-flag  { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-family: 'Space Mono', monospace; }
    .check-clear { background: rgba(34,197,94,0.15);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.3);  border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-family: 'Space Mono', monospace; }

    .rec-box { background: #111520; border: 1px solid #2a3356; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.9rem; color: #e8eaf6; line-height: 1.7; margin-top: 0.5rem; }
    .summary-box { border-left: 2px solid #2a3356; padding-left: 1rem; color: #8892b0; font-style: italic; font-size: 0.88rem; line-height: 1.7; margin: 0.5rem 0 1rem; }

    hr { border-color: #1e2540 !important; }

    .stTextArea textarea { background: #111520 !important; border: 1px solid #2a3356 !important; border-radius: 10px !important; color: #e8eaf6 !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 0.9rem !important; }
    .stTextArea textarea:focus { border-color: #6c63ff !important; }
    .stButton button { border-radius: 8px !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load HuggingFace model (cached so it only downloads once) ─────────────────
@st.cache_resource(show_spinner="Loading AI model (first time only, ~1 min)…")
def load_model():
    return pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification",
        truncation=True,
        max_length=512,
        return_all_scores=True  # get scores for ALL labels so we can compare them
    )

# ── Rule-based signal detector ─────────────────────────────────────────────────
def detect_signals(text):
    signals = []
    checklist = []

    # 1. Sensationalist language
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim    = text.count('!') + text.count('?!')
    flagged_caps = caps_ratio > 0.25 or exclaim >= 2
    checklist.append({"label": "Sensationalist language (ALL CAPS, excessive punctuation)", "flagged": flagged_caps})
    if flagged_caps:
        signals.append({"label": "ALL CAPS / Over-punctuation", "risk": "high"})

    # 2. Emotional manipulation
    emotional_words = ["shocking", "unbelievable", "outrage", "panic", "terrifying",
                       "destroy", "exposed", "evil", "disgusting", "horrifying", "bombshell"]
    flagged_emotion = any(w in text.lower() for w in emotional_words)
    checklist.append({"label": "Emotional manipulation or fear/outrage bait", "flagged": flagged_emotion})
    if flagged_emotion:
        signals.append({"label": "Emotional Manipulation", "risk": "high"})

    # 3. Vague sources
    vague_sources = ["experts say", "scientists claim", "sources say", "according to insiders",
                     "some people say", "many believe", "everyone knows", "they say"]
    flagged_sources = any(p in text.lower() for p in vague_sources)
    checklist.append({"label": "Vague or unnamed sources ('experts say', 'scientists claim')", "flagged": flagged_sources})
    if flagged_sources:
        signals.append({"label": "Unnamed Sources", "risk": "medium"})

    # 4. Urgency to share
    urgency_phrases = ["share before", "deleted soon", "they don't want you", "spread the word",
                       "before it's too late", "share now", "pass this on", "forward this"]
    flagged_urgency = any(p in text.lower() for p in urgency_phrases)
    checklist.append({"label": "Urgency to share before deletion", "flagged": flagged_urgency})
    if flagged_urgency:
        signals.append({"label": "Share Urgency Pressure", "risk": "high"})

    # 5. Unverifiable claims
    unverifiable = ["100%", "guaranteed", "proven fact", "confirmed by", "scientists confirm",
                    "studies show", "research proves", "experts agree"]
    flagged_unverif = any(p in text.lower() for p in unverifiable)
    checklist.append({"label": "Unverifiable statistics or events", "flagged": flagged_unverif})
    if flagged_unverif:
        signals.append({"label": "Unverifiable Claims", "risk": "medium"})

    # Credibility indicator (named org, date, numbers)
    has_named_org = bool(re.search(r'\b(WHO|UN|CDC|Reuters|BBC|AP|NASA|FDA|government|ministry|university)\b', text, re.I))
    has_date      = bool(re.search(r'\b(20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b', text, re.I))
    if has_named_org or has_date:
        signals.append({"label": "Named Source / Date Present", "risk": "low"})

    return signals, checklist


def get_recommendation(verdict, checklist):
    flagged = [c["label"] for c in checklist if c["flagged"]]
    if verdict == "FAKE":
        if any("urgency" in f.lower() for f in flagged):
            return "Do NOT share this content. Urgency-to-share is a classic manipulation tactic used to spread misinformation quickly."
        return "Cross-check this claim on fact-checking sites like Snopes, FactCheck.org, or Reuters Fact Check before believing or sharing."
    elif verdict == "REAL":
        return "This appears credible, but always verify with the original source directly before sharing."
    else:
        return "This content has mixed signals. Search for the original source and check if reputable outlets are reporting the same story."


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🔍 VERIFAI</h1>
  <p>AI-powered fake news detection · RoBERTa model · No API key needed</p>
</div>
""", unsafe_allow_html=True)

# ── Examples ──────────────────────────────────────────────────────────────────
EXAMPLES = {
    "🔴 Likely Fake": "BREAKING: Scientists CONFIRM 5G towers are secretly altering human DNA — Share before this gets DELETED!!",
    "🟢 Likely Real": "WHO reports 4.2% rise in global malaria cases in 2023, citing disruptions to health services in sub-Saharan Africa.",
    "🟡 Uncertain":   "New study finds coffee drinkers live longer — experts say it could add up to 5 years to your life."
}

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (label, text) in enumerate(EXAMPLES.items()):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"ex_{i}"):
            st.session_state.input_text = text

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem">Paste news content</div>', unsafe_allow_html=True)
news_input = st.text_area(
    label="",
    value=st.session_state.get("input_text", ""),
    height=160,
    placeholder="Paste a headline, article excerpt, or social media post here…",
    max_chars=2000,
    label_visibility="collapsed"
)

# ── Analyze ───────────────────────────────────────────────────────────────────
if st.button("🔎 Analyze Content", type="primary", disabled=len(news_input.strip()) < 10, use_container_width=True):
    with st.spinner("Running AI analysis…"):
        try:
            classifier = load_model()
            # return_all_scores=True gives us a list of {label, score} for every class
            all_scores = classifier(news_input[:512])[0]

            # Build a clean dict: {"FAKE": 0.87, "REAL": 0.13} regardless of label naming
            score_map = {}
            for item in all_scores:
                lbl = item["label"].upper()
                # hamzab model uses LABEL_0=FAKE, LABEL_1=REAL OR direct FAKE/REAL strings
                if lbl in ("LABEL_0", "FAKE"):
                    score_map["FAKE"] = item["score"]
                elif lbl in ("LABEL_1", "REAL"):
                    score_map["REAL"] = item["score"]

            fake_score = score_map.get("FAKE", 0.5)
            real_score = score_map.get("REAL", 0.5)
            margin     = abs(fake_score - real_score)

            if margin < 0.15:
                # Model is not confident enough — genuinely uncertain
                verdict    = "UNCERTAIN"
                confidence = int((0.5 + margin / 2) * 100)
            elif fake_score > real_score:
                verdict    = "FAKE"
                confidence = int(fake_score * 100)
            else:
                verdict    = "REAL"
                confidence = int(real_score * 100)

            signals, checklist = detect_signals(news_input)
            recommendation     = get_recommendation(verdict, checklist)

            vc   = verdict.lower()
            icon = {"real": "✅", "fake": "❌", "uncertain": "⚠️"}.get(vc, "⚠️")

            st.divider()

            # Verdict
            st.markdown(f"""
            <div class="verdict-card verdict-{vc}">
              <div class="verdict-title">Verdict</div>
              <div class="verdict-value-{vc}">{icon} {verdict}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence
            st.markdown('<div class="section-label">Model Confidence</div>', unsafe_allow_html=True)
            st.progress(confidence / 100, text=f"{confidence}%")

            # Signals
            if signals:
                st.markdown('<div class="section-label">Detection Signals</div>', unsafe_allow_html=True)
                risk_dot   = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                badges_html = '<div class="signals-row">' + "".join(
                    f'<span class="badge badge-{s["risk"]}">{risk_dot.get(s["risk"],"")} {s["label"]}</span>'
                    for s in signals
                ) + '</div>'
                st.markdown(badges_html, unsafe_allow_html=True)

            # Checklist
            st.markdown('<div class="section-label">Misinformation Checklist</div>', unsafe_allow_html=True)
            for item in checklist:
                tag = '<span class="check-flag">FLAGGED</span>' if item["flagged"] else '<span class="check-clear">CLEAR</span>'
                st.markdown(f'<div class="check-row">{tag} {item["label"]}</div>', unsafe_allow_html=True)

            # Recommendation
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Recommendation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rec-box">💡 {recommendation}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 If the model failed to download, check your internet connection and try again.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center; color:#3d4470; font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.08em;">
    VERIFAI · Built with Streamlit + HuggingFace RoBERTa · Snehalata Sethi
</p>
""", unsafe_allow_html=True)
