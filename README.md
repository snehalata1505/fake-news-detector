# 🔍 Verifai — Fake News Detector

AI-powered media credibility analysis using **HuggingFace RoBERTa** + **Streamlit**.
> ✅ No API key needed. Completely free. Works offline after first model download.

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Snehalata-Sethi/fake-news-detector.git
cd fake-news-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

> First run will download the RoBERTa model (~500MB). After that it's cached and instant.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select repo → set main file as `app.py`
4. Click **Deploy** — no secrets needed! ✅

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Frontend | Streamlit + custom CSS |
| AI Model | RoBERTa (hamzab/roberta-fake-news-classification) via HuggingFace |
| Signal Detection | Rule-based Python (caps, urgency, sources, emotion) |
| Language | Python 3.9+ |
| Deployment | Streamlit Cloud via GitHub |

---

## ✨ Features

- **Verdict** — REAL / FAKE / UNCERTAIN with confidence score
- **Detection Signals** — color-coded risk badges
- **Misinformation Checklist** — 5 key red flag checks
- **Recommendation** — one actionable step for the reader
- **No API key required** — uses free HuggingFace model

---

Built by [Snehalata Sethi](https://github.com/Snehalata-Sethi)
