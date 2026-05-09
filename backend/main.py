"""
GovScheme Advisor — FastAPI Main Application
NLP-powered Government Scheme Recommendation System for India
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from database import connect_db, close_db, get_db
from api.routes import router
from seed_data import SCHEMES
from nlp.engine import nlp_engine
from nlp.analysis import policy_analyzer
from nlp.trainer import train_and_save, load_model, MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("\n🏛️  GovScheme Advisor — Starting up...")
    
    # Connect to database
    db = await connect_db()
    
    # Seed schemes if DB is empty
    count = await db.schemes.count_documents({})
    if count == 0:
        print("  📦 Seeding database with government schemes...")
        for scheme in SCHEMES:
            scheme["status"] = "active"
            scheme["created_at"] = datetime.utcnow()
            scheme["updated_at"] = datetime.utcnow()
            try:
                await db.schemes.insert_one(scheme)
            except Exception:
                pass
        print(f"  ✅ Seeded {len(SCHEMES)} schemes")
    else:
        print(f"  ✅ Found {count} schemes in database")
    
    # Create default admin
    admin_email = os.getenv("ADMIN_EMAIL", "admin@govscheme.in")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    existing_admin = await db.users.find_one({"email": admin_email})
    if not existing_admin:
        from passlib.context import CryptContext
        pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
        await db.users.insert_one({
            "email": admin_email,
            "hashed_password": pwd.hash(admin_pass),
            "role": "admin",
            "created_at": datetime.utcnow()
        })
        print(f"  ✅ Admin user created: {admin_email}")
    
    # Pre-load NLP engine
    schemes_list = []
    async for s in db.schemes.find({"status": "active"}):
        s.pop("_id", None)
        schemes_list.append(s)
    nlp_engine.load_schemes(schemes_list)
    print(f"  ✅ NLP engine loaded with {len(schemes_list)} schemes")
    
    # Auto-train classifier on first boot if pkl doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("  🧠 Training NLP classifier for the first time...")
        metrics = train_and_save(schemes_list)
        print(f"  ✅ NLP classifier trained and saved. "
              f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score_macro']}, "
              f"Samples: {metrics['num_samples']}")
        nlp_engine.reload_classifier()
    else:
        if nlp_engine.classifier_available:
            print("  ✅ NLP classifier loaded from disk.")
        else:
            print("  ⚠️  Classifier pkl exists but failed to load.")

    # Fit the policy analyzer (LDA topics, semantic clusters, keywords, sentiment)
    print("  🧬 Fitting policy analyzer (LDA + semantic topics + sentiment)...")
    try:
        analysis_summary = policy_analyzer.fit(schemes_list)
        print(
            f"  ✅ Analyzer ready — {analysis_summary['n_schemes']} schemes, "
            f"{analysis_summary['n_topics_lda']} LDA topics, "
            f"{analysis_summary['n_topics_semantic']} semantic topics "
            f"(backend: {analysis_summary['semantic_backend']})."
        )
    except Exception as exc:
        print(f"  ⚠️  Analyzer fit failed: {exc}")

    print("  🚀 Server ready!\n")
    
    yield
    
    await close_db()
    print("\n👋 GovScheme Advisor — Shutdown complete")


app = FastAPI(
    title="GovScheme Advisor API",
    description="NLP-powered Government Scheme Recommendation System for India",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "GovScheme Advisor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    db = get_db()
    scheme_count = await db.schemes.count_documents({}) if db is not None else 0
    return {"status": "healthy", "schemes_loaded": scheme_count}
