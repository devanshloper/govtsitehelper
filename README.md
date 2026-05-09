# 🏛️ GovScheme Advisor - NLP Recommendation Chatbot

GovScheme Advisor is a complete full-stack web application that uses Natural Language Processing to match Indian citizens with government schemes they are eligible for. Instead of a standard chat interface, users fill out a structured form (grid layout) with their details, and the backend engine calculates eligibility using a hybrid approach.

## 🚀 Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, React Router DOM, Lucide Icons.
- **Backend**: Python FastAPI.
- **Database**: MongoDB (via Motor async driver).
- **NLP Engine**: `scikit-learn` (TF-IDF Vectorization & Cosine Similarity) + Rule-Based Engine + Trainable ML Classifier.
- **ML Classifier**: TF-IDF → OneVsRest Logistic Regression pipeline (trained on synthetic demographic data).
- **Web Scraper**: `BeautifulSoup`, `Requests`.

## 🧠 Recommendation Logic & Training Data

The NLP engine is **not** a large pre-trained LLM (like GPT or Llama). Instead, it uses a highly efficient, deterministic hybrid approach to rank schemes:

1. **Rule-Based Filters**: The system strictly checks hard constraints. If a scheme requires a female applicant, it explicitly filters out males. It checks exact parameters like:
   - Age limits
   - Income limits (e.g., `< 2.5 Lakhs/yr`)
   - Caste categories
   - BPL status
   - Specific tags (Widow, Student, Farmer, etc.)

2. **Semantic Matching (TF-IDF)**: After rule-based filtering, the engine converts the user's profile into a descriptive natural language string. It uses a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) trained dynamically on the loaded scheme dataset to calculate **Cosine Similarity** between the user's profile text and the scheme's text (description, eligibility text, benefits, and tags).

3. **Trainable ML Classifier**: A `scikit-learn` pipeline (TF-IDF + OneVsRest Logistic Regression) trained on synthetic user profile data generated from scheme eligibility rules. It provides a classifier confidence boost to the final score. The model is persisted as a `.pkl` file and hot-loaded at server startup.

4. **Scoring System**: The final rank is calculated using a weighted formula:
   - **With classifier**: `(0.4 × rule_score) + (0.4 × tfidf_score) + (0.2 × classifier_boost)`
   - **Without classifier**: `(0.5 × rule_score) + (0.35 × tfidf_score) + (0.15 × priority_boost)`

### What data is it trained/seeded on?

The system contains **243 real Indian Government Schemes**. It is initially seeded with 25 manually curated schemes, and further expanded using a custom intelligent web scraper that pulls data from Wikipedia. 

Some of the seeded schemes include:
- PM-KISAN Samman Nidhi
- Pradhan Mantri Awas Yojana (PMAY)
- Pradhan Mantri Jan Dhan Yojana (PMJDY)
- Ayushman Bharat PM-JAY
- Mahatma Gandhi National Rural Employment Guarantee (MGNREGA)
- Pradhan Mantri Ujjwala Yojana
- Various Post Matric Scholarships (SC/ST/OBC/Minority)
- Sukanya Samriddhi Yojana
- MUDRA Loan Yojana

Additionally, the project includes an **Intelligent NLP Web Scraper** (`backend/scraper/scraper.py`) that:
- Scrapes the comprehensive "List of government schemes in India" from Wikipedia.
- Visits individual scheme pages to extract descriptions and benefits.
- Uses NLP keyword analysis to **auto-categorize** schemes and **extract eligibility rules** (age, gender, BPL status, etc.).
- Automatically structures and inserts the scraped schemes into MongoDB.

The system also features an **Admin Dashboard** (available at `/admin` on the frontend) where you can view live database analytics, check the NLP model status, and trigger the Wikipedia scraper to refresh the database with a single click.

## 🧬 Policy Analysis Dashboard (`/analysis`)

A second NLP pipeline runs alongside the recommender and powers a dedicated dashboard at `/analysis` on the frontend. It treats the scheme corpus as a policy-document collection and produces:

| Capability | Method | Surface |
|---|---|---|
| **Topic modeling** | `sklearn` Latent Dirichlet Allocation over CountVectorizer features | Topic explorer with per-topic top words and member schemes |
| **Semantic / "BERTopic"-style topics** | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) clustered with K-Means; falls back to TF-IDF + K-Means when `sentence-transformers` is not installed | Toggle in the topic explorer |
| **Keyword extraction** | TF-IDF top-N per scheme, aggregated into a corpus-wide cloud | Keyword cloud + per-scheme keyword lists |
| **Sentiment / policy tone** | Lexicon-based (positive/negative wordlists tuned for policy text), no external corpus download | Tone breakdown + sentiment-by-state and sentiment-by-category panels |
| **Cross-cuts** | Aggregations over `state` and `category` fields | Distribution and polarity panels |

The analyzer is **fitted once at FastAPI startup** (`nlp.analysis.policy_analyzer.fit(schemes)`) and served from cached state through `/api/analysis/*`. For the static GitHub Pages build, `python backend/generate_static_analysis.py` materialises the same bundle to `frontend/public/analysis.json`; the Analysis page tries the API first and falls back to that file automatically.

### Analysis API

| Endpoint | Returns |
|---|---|
| `GET /api/analysis/overview` | Full bundle (used by the dashboard) |
| `GET /api/analysis/summary` | Counts and sentiment breakdown |
| `GET /api/analysis/topics?kind=lda\|semantic` | Topic list with top words and member counts |
| `GET /api/analysis/topic/{id}?kind=lda\|semantic` | Schemes assigned to a given topic |
| `GET /api/analysis/keywords[?scheme_id=]` | Keyword cloud or per-scheme keywords |
| `GET /api/analysis/sentiment` | Corpus, by-state, and by-category sentiment |
| `GET /api/analysis/comparison` | Distribution and polarity matrices |
| `GET /api/analysis/scheme/{id}` | Per-scheme keywords, sentiment, assigned topics |
| `POST /api/admin/analysis/refit` | Refit the analyzer on the current DB (admin only) |

## 📊 Evaluation Metrics

The system was evaluated across three categories to measure ML accuracy, recommendation quality, and system performance.

### 1. ML Classification Metrics (5-Fold Cross-Validation)

| Metric | Value |
|---|---|
| **Macro F1-Score** | 0.3587 (±0.0141) |
| **Macro Precision** | 0.3758 (±0.0249) |
| **Macro Recall** | 0.3907 (±0.0118) |
| **Accuracy** | 0.3876 (±0.0085) |
| Training Samples | 436 |
| Classes (Schemes) | 25 |
| CV Folds | 5 |

> **Note:** The classifier operates on 25-class synthetic data with ~17 samples per class. Schemes with distinctive eligibility features achieve near-perfect classification (e.g., `disability-pension` F1=1.00, `widow-pension` F1=1.00, `minority-scholarship` F1=0.97). Generic schemes with broad eligibility overlap are harder to distinguish. The classifier serves as a **supplementary signal** — the primary ranking comes from rule-based filtering and TF-IDF semantic matching.

### 2. Recommendation Engine Metrics

| Metric | Value |
|---|---|
| **Precision@5** | 0.6200 |
| **Mean Reciprocal Rank (MRR)** | 0.8683 |
| Profiles Evaluated | 20 |

Evaluated using 20 diverse mock user profiles (varying age, income, caste, gender, and tags like student/farmer/BPL) with manually defined ground-truth expected schemes.

### 3. System Performance & API Latency

| Metric | Value |
|---|---|
| **Avg End-to-End API Latency** | 17.30 ms |
| **Median Latency** | 15.91 ms |
| **P95 Latency** | 24.40 ms |
| Min / Max Latency | 15.21 / 45.27 ms |
| Requests (50/50 successful) | 100% |

**Server-Side Breakdown (avg):**

| Component | Latency |
|---|---|
| DB Fetch (MongoDB) | 1.33 ms |
| TF-IDF Indexing | 2.34 ms |
| Recommendation Scoring | 5.44 ms |
| Server Total | 9.12 ms |

### Evaluation Scripts

| Script | Purpose |
|---|---|
| `nlp_pipeline.py --evaluate` | 5-fold CV classification metrics |
| `backend/test_recsys_metrics.py` | Precision@5 & MRR evaluation |
| `backend/test_latency.py` | 50-request API latency benchmark |

Results are saved to:
- `backend/metrics_classification_results.txt`
- `backend/metrics_recsys_results.txt`
- `backend/metrics_latency_results.txt`

## 📦 How to Run

1. Ensure you have **MongoDB** running locally on port `27017` (default).
2. Ensure you have **Node.js** and **Python 3.9+** installed.
3. Run the startup script from the root directory:

```bash
chmod +x start.sh
./start.sh
```

This script will automatically:
- Create a Python virtual environment and install backend dependencies (`pip install -r requirements.txt`).
- Install React dependencies (`npm install`).
- Boot the FastAPI backend on `http://localhost:8000`.
- Boot the Vite React frontend on `http://localhost:5173`.

### Running Evaluation Scripts

```bash
# Train classifier and generate classification report
python nlp_pipeline.py --train

# Run 5-fold cross-validation
python nlp_pipeline.py --evaluate

# Run recommendation engine metrics (standalone, no server needed)
cd backend && python test_recsys_metrics.py

# Run API latency benchmark (requires server running on port 8000)
cd backend && python test_latency.py
```

## 📂 Project Structure

```
govtsitehelper/
├── start.sh                      # Unified startup script
├── nlp_pipeline.py               # CLI: train, evaluate, predict classifier
├── backend/                      # FastAPI Backend
│   ├── main.py                   # App entrypoint & Lifespan events
│   ├── database.py               # MongoDB connection
│   ├── models.py                 # Pydantic schemas
│   ├── seed_data.py              # 25 real Government Schemes
│   ├── nlp/
│   │   ├── engine.py             # Rule-based + TF-IDF + Classifier logic
│   │   ├── trainer.py            # ML classifier training & persistence
│   │   ├── analysis.py           # LDA + semantic topics + keywords + sentiment
│   │   └── models/               # Saved .pkl model files
│   ├── api/
│   │   └── routes.py             # API Endpoints (Auth, Recommend, Search, Analysis, Admin)
│   ├── generate_static_analysis.py  # Emits frontend/public/analysis.json for static hosting
│   ├── scraper/
│   │   └── scraper.py            # Web Scraper (BeautifulSoup)
│   ├── test_recsys_metrics.py    # Evaluation: Precision@5 & MRR
│   ├── test_latency.py           # Evaluation: API latency benchmark
│   ├── metrics_classification_results.txt
│   ├── metrics_recsys_results.txt
│   └── metrics_latency_results.txt
└── frontend/                     # React Frontend
    ├── index.html
    ├── vite.config.js            # Vite config with Proxy to Backend
    └── src/
        ├── App.jsx               # Routing
        ├── index.css             # Tailwind setup & theme
        ├── components/
        │   └── Navbar.jsx        # Navigation bar
        ├── pages/
        │   ├── Home.jsx          # Multi-column form UI
        │   ├── Search.jsx        # NLP Semantic Search UI
        │   ├── Analysis.jsx      # Policy analysis dashboard (LDA + sentiment + keywords)
        │   ├── Dashboard.jsx     # Result ranking & details UI
        │   └── Admin.jsx         # Admin Dashboard for DB & Scraper management
        └── public/
            └── analysis.json     # Pre-computed analysis bundle for static hosting
```
