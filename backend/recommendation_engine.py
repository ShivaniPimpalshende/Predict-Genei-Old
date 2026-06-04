import pandas as pd
from sqlalchemy.orm import Session
from models import MarketingData
from typing import Dict, Any
from prediction_engine import PredictionEngine
import json
import os
from dotenv import load_dotenv

# ✅ NEW GEMINI SDK
from google import genai

load_dotenv()

# ✅ AI Studio API key (NOT Google Cloud restricted key)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class RecommendationEngine:
    """AI-powered recommendation engine with Gemini captions"""

    @staticmethod
    def generate_recommendations(db: Session) -> Dict[str, Any]:

        data = db.query(MarketingData).all()

        if not data:
            return {
                "best_posting_times": [],
                "best_content_types": [],
                "best_cta": [],
                "caption_suggestions": [],
                "overall_insights": [
                    "Not enough data available. Please upload more marketing data."
                ]
            }

        # -------------------------
        # DataFrame
        # -------------------------
        df = pd.DataFrame([{
            "post_type": d.post_type,
            "hour": d.created_at.hour if d.created_at else 0,
            "follower_count": d.follower_count,
            "hashtag_count": d.hashtag_count,
            "mention_count": d.mention_count,
            "CTA_used": d.CTA_used if d.CTA_used else "no_cta",
            "engagement_score": RecommendationEngine.calculate_engagement(
                d.like_count, d.comment_count, d.repost_count, d.follower_count
            )
        } for d in data])

        # -------------------------
        # 📊 TIME ANALYSIS
        # -------------------------
        time_performance = (
            df.groupby("hour")["engagement_score"]
            .mean()
            .sort_values(ascending=False)
        )

        best_hours = time_performance.head(3).index.tolist()

        best_times = [
            {
                "hour": int(hour),
                "time_label": RecommendationEngine._format_time(hour),
                "avg_engagement": round(time_performance[hour] * 100, 2)
            }
            for hour in best_hours
        ]

        # -------------------------
        # 🤖 CONTENT TYPE (ML)
        # -------------------------
        prediction_engine = PredictionEngine()

        if not prediction_engine.is_trained:
            return {"error": "Model not trained. Train model first."}

        avg_followers = int(df["follower_count"].mean())
        unique_post_types = df["post_type"].unique()

        predictions = []

        for post_type in unique_post_types:
            result = prediction_engine.predict(
                follower_count=avg_followers,
                post_type=post_type,
                hashtag_count=int(df["hashtag_count"].mean()),
                mention_count=int(df["mention_count"].mean()),
                cta_used="no_cta"
            )
            print("Prediction Result:", result)
            if result["success"]:
                predictions.append({
                    "content_type": post_type,
                    "avg_engagement": round(result["predicted_engagement_score"] * 100, 2),
                     "confidence_score": result["confidence_score"],
                    "recommendation": f"{post_type.capitalize()} performs best"
                     
                })

        best_content = sorted(
            predictions,
            key=lambda x: x["avg_engagement"],
            reverse=True
        )[:3]

        # -------------------------
        # 🤖 CTA ANALYSIS
        # -------------------------
        cta_options = df["CTA_used"].unique().tolist()

        if len(cta_options) < 3:
            cta_options += ["buy_now", "learn_more", "subscribe", "shop_now"]

        cta_options = list(set(cta_options))

        avg_hashtags = int(df["hashtag_count"].mean())
        avg_mentions = int(df["mention_count"].mean())
        best_post_type = df["post_type"].mode()[0]

        cta_predictions = []

        for cta in cta_options:
            result = prediction_engine.predict(
                follower_count=avg_followers,
                post_type=best_post_type,
                hashtag_count=avg_hashtags,
                mention_count=avg_mentions,
                cta_used=cta
            )

            if result["success"]:
                cta_predictions.append({
                    "cta": cta,
                    "predicted_engagement": round(result["predicted_engagement_score"] * 100, 2),
                     "confidence_score": result["confidence_score"]
                })

        best_cta = sorted(
            cta_predictions,
            key=lambda x: x["predicted_engagement"],
            reverse=True
        )[:2]

        # -------------------------
        # 🔥 GEMINI CAPTIONS (NEW SDK)
        # -------------------------
        caption_suggestions = RecommendationEngine._generate_llm_captions(df)

        # -------------------------
        # 📊 INSIGHTS
        # -------------------------
        insights = RecommendationEngine._generate_insights(df)

        return {
            "avg_engagement": round(df["engagement_score"].mean() * 100, 2),
            "best_posting_times": best_times,
            "best_content_types": best_content,
            "best_cta": best_cta,
            "caption_suggestions": caption_suggestions,
            "overall_insights": insights
        }

    # =========================
    # 🔥 GEMINI CAPTION GENERATOR (FIXED)
    # =========================
    @staticmethod
    def _generate_llm_captions(df: pd.DataFrame):

        best_post_type = df["post_type"].mode()[0]
        best_cta = df.groupby("CTA_used")["engagement_score"].mean().idxmax()
        avg_engagement = round(df["engagement_score"].mean(), 4)
        avg_hashtags = int(df["hashtag_count"].mean())

        prompt = f"""
You are a professional social media marketing expert.

Generate 5 short caption tips.

Context:
- Best content type: {best_post_type}
- Best CTA: {best_cta}
- Avg engagement: {avg_engagement}
- Avg hashtags: {avg_hashtags}

Rules:
- Each tip must be 5–10 words
- Actionable advice only

Return ONLY JSON:
[
  {{ "caption": "..." }},
  {{ "caption": "..." }},
  {{ "caption": "..." }},
  {{ "caption": "..." }},
  {{ "caption": "..." }}
]
"""

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )

            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except Exception as e:
            print("Gemini Error:", e)

            return [
                {"caption": "Ask questions to boost engagement"},
                {"caption": "Use strong CTA like follow or share"},
                {"caption": "Keep captions short and clear"},
                {"caption": "Use 5–7 relevant hashtags"},
                {"caption": "Post consistently for better growth"}
            ]

    # =========================
    # 📊 ENGAGEMENT CALCULATION
    # =========================
    @staticmethod
    def calculate_engagement(like_count, comment_count, repost_count, follower_count):
        return (like_count + 2 * comment_count + 3 * repost_count) / max(follower_count, 1)

    # =========================
    # ⏰ FORMAT TIME
    # =========================
    @staticmethod
    # def _format_time(hour: int) -> str:
    #     period = "AM" if hour < 12 else "PM"
    #     display_hour = hour if hour <= 12 else hour - 12
    #     if display_hour == 0:
    #         display_hour = 12
    #     return f"{display_hour}:00 {period}"
    @staticmethod
    def _format_time(hour: int) -> str:
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:00 PM"
    # =========================
    # 📊 INSIGHTS
    # =========================
    @staticmethod
    def _generate_insights(df: pd.DataFrame):

        insights = []

        avg_engagement = round(df["engagement_score"].mean() * 100, 2)
        insights.append(f"Average engagement score: {avg_engagement}")

        best_content = df.groupby("post_type")["engagement_score"].mean().idxmax()
        insights.append(f"{best_content.capitalize()} posts performed best")

        best_hour = df.groupby("hour")["engagement_score"].mean().idxmax()
        insights.append(f"Best posting time: {RecommendationEngine._format_time(best_hour)}")

        avg_hashtags = round(df["hashtag_count"].mean(), 1)
        insights.append(f"Average hashtags used: {avg_hashtags}")

        return insights