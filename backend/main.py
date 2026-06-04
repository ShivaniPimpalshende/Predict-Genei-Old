# from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from sqlalchemy.orm import Session
# import pandas as pd
# import io
# import os

# from database import get_db, init_db
# from models import MarketingData
# from schemas import (
#     MarketingDataInput,
#     MarketingDataResponse,
#     PredictionInput,
#     PredictionResponse
# )
# from analytics_engine import AnalyticsEngine
# from prediction_engine import PredictionEngine
# from recommendation_engine import RecommendationEngine

# # -----------------------------
# # FastAPI App
# # -----------------------------
# app = FastAPI(title="Predict Genie", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# prediction_engine = PredictionEngine()

# # -----------------------------
# # STARTUP
# # -----------------------------
# @app.on_event("startup")
# def startup_event():
#     init_db()

#     # ✅ AUTO LOAD TRAINED MODEL
#     if prediction_engine.is_trained:
#         print("✅ Model Loaded Successfully")
#     else:
#         print("⚠️ Model not found. Train once using script")

# # -----------------------------
# # HEALTH
# # -----------------------------
# @app.get("/")
# def root():
#     return {"message": "API running"}

# # -----------------------------
# # CSV UPLOAD (USER TEST DATA)
# # -----------------------------
# @app.post("/upload-data")
# async def upload_data(file: UploadFile = File(...), db: Session = Depends(get_db)):

#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Upload CSV only")

#     db.query(MarketingData).delete()
#     db.commit()

#     contents = await file.read()
#     df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
#     df.columns = df.columns.str.strip().str.lower()

#     required_columns = [
#         "name",
#         "follower_count",
#         "post_type",
#         "like_count",
#         "comment_count",
#         "repost_count",
#         "hashtag_count",
#         "mention_count",
#         "cta_used"
#     ]

#     missing = [c for c in required_columns if c not in df.columns]
#     if missing:
#         raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

#     df["cta_used"] = df["cta_used"].fillna("no_cta")

#     for _, row in df.iterrows():
#         record = MarketingData(
#             name=str(row["name"]),
#             follower_count=int(row["follower_count"]),
#             post_type=str(row["post_type"]).lower(),
#             like_count=int(row["like_count"]),
#             comment_count=int(row["comment_count"]),
#             repost_count=int(row["repost_count"]),
#             hashtag_count=int(row["hashtag_count"]),
#             mention_count=int(row["mention_count"]),
#             CTA_used=str(row["cta_used"])
#         )

#         record.engagement_score = (
#             record.like_count +
#             2 * record.comment_count +
#             3 * record.repost_count
#         ) / max(record.follower_count, 1)

#         db.add(record)

#     db.commit()

#     return {"success": True, "message": "User CSV uploaded"}

# # -----------------------------
# # ANALYTICS (USER DATA)
# # -----------------------------
# @app.get("/analytics")
# def analytics(db: Session = Depends(get_db)):
#     return AnalyticsEngine.get_analytics(db)

# # -----------------------------
# # PREDICT (MODEL USE)
# # -----------------------------
# @app.post("/predict", response_model=PredictionResponse)
# def predict(data: PredictionInput):

#     result = prediction_engine.predict(
#         follower_count=data.follower_count,
#         post_type=data.post_type,
#         hashtag_count=data.hashtag_count,
#         mention_count=data.mention_count,
#         cta_used=data.cta_used
#     )

#     if not result["success"]:
#         raise HTTPException(status_code=400, detail=result["message"])

#     return result

# # -----------------------------
# # RECOMMENDATIONS (USER DATA)
# # -----------------------------
# @app.get("/recommendations")
# def recommendations(db: Session = Depends(get_db)):
#     return RecommendationEngine.generate_recommendations(db)

# @app.get("/data/count")
# def get_data_count(db: Session = Depends(get_db)):
#     count = db.query(MarketingData).count()
#     return {"count": count}

# @app.get("/data/recent")
# def get_recent_data(limit: int = 10, db: Session = Depends(get_db)):
#     data = db.query(MarketingData).order_by(MarketingData.id.desc()).limit(limit).all()
#     return data
# @app.get("/analytics/post-type")
# def post_type_analysis(db: Session = Depends(get_db)):
#     return AnalyticsEngine.get_post_type_analysis(db)

# @app.get("/analytics/time-analysis")
# def time_analysis(db: Session = Depends(get_db)):
#     return AnalyticsEngine.get_time_analysis(db)
# # -----------------------------
# # RUN
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", reload=True)

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import io
import random
from datetime import datetime

from database import get_db, init_db
from models import MarketingData
from schemas import PredictionInput, PredictionResponse
from analytics_engine import AnalyticsEngine
from prediction_engine import PredictionEngine
from recommendation_engine import RecommendationEngine

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Predict Genie", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_engine = PredictionEngine()

# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ DB Ready")

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"message": "API running"}

# -----------------------------
# CSV UPLOAD (FINAL FIXED)
# -----------------------------
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...), db: Session = Depends(get_db)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV only")

    # 🔥 CLEAR OLD DATA
    db.query(MarketingData).delete()
    db.commit()

    # READ CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "name",
        "follower_count",
        "post_type",
        "like_count",
        "comment_count",
        "repost_count",
        "hashtag_count",
        "mention_count",
        "cta_used"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # FIX NULL VALUES
    df["cta_used"] = df["cta_used"].fillna("no_cta")
    df = df.fillna(0)

    # 🎯 RANDOM TIME PER UPLOAD (ONLY ONCE)
    random_hour = random.randint(0, 23)

    base_time = datetime.now().replace(
        hour=random_hour,
        minute=0,
        second=0,
        microsecond=0
    )

    records_added = 0

    # LOOP
    for _, row in df.iterrows():

        record = MarketingData(
            name=str(row["name"]),
            follower_count=int(row["follower_count"]),
            post_type=str(row["post_type"]).lower(),
            like_count=int(row["like_count"]),
            comment_count=int(row["comment_count"]),
            repost_count=int(row["repost_count"]),
            hashtag_count=int(row["hashtag_count"]),
            mention_count=int(row["mention_count"]),
            CTA_used=str(row["cta_used"]),
            created_at=base_time   # ✅ SAME TIME FOR WHOLE CSV
        )

        # ✅ CORRECT ENGAGEMENT FORMULA
        record.engagement_score = (
            record.like_count +
            2 * record.comment_count +
            3 * record.repost_count
        ) / max(record.follower_count, 1)

        db.add(record)
        records_added += 1

    db.commit()

    return {
        "success": True,
        "records_added": records_added,
        "total_rows": len(df),
        "upload_hour": random_hour  # debug (you can remove later)
    }

# -----------------------------
# ANALYTICS
# -----------------------------
@app.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    return AnalyticsEngine.get_analytics(db)

# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):

    result = prediction_engine.predict(
        follower_count=data.follower_count,
        post_type=data.post_type,
        hashtag_count=data.hashtag_count,
        mention_count=data.mention_count,
        cta_used=data.cta_used
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return result

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
@app.get("/recommendations")
def recommendations(db: Session = Depends(get_db)):
    return RecommendationEngine.generate_recommendations(db)

# -----------------------------
# DEBUG APIs
# -----------------------------
@app.get("/data/count")
def get_data_count(db: Session = Depends(get_db)):
    return {"count": db.query(MarketingData).count()}

@app.get("/data/recent")
def get_recent_data(db: Session = Depends(get_db)):
    return db.query(MarketingData).all()

# -----------------------------
# EXTRA ANALYTICS
# -----------------------------
@app.get("/analytics/post-type")
def post_type_analysis(db: Session = Depends(get_db)):
    return AnalyticsEngine.get_post_type_analysis(db)

@app.get("/analytics/time-analysis")
def time_analysis(db: Session = Depends(get_db)):
    return AnalyticsEngine.get_time_analysis(db)

# -----------------------------
# TRAIN MODEL
# -----------------------------
@app.post("/train-model")
def train_model(db: Session = Depends(get_db)):

    result = prediction_engine.train(db)

    return result
# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)