

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, Any
from sqlalchemy.orm import Session


class PredictionEngine:
    """ML-based prediction engine for engagement forecasting"""

    def __init__(self, model_path: str = "models/engagement_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.load_model()

    # =========================
    # TRAIN MODEL (FROM CSV ONLY)
    # =========================
    def train_model(self, db: Session) -> Dict[str, Any]:
        try:
            # ✅ Prevent retraining
            if self.is_trained:
                return {
                    "success": True,
                    "message": "Model already trained (loaded from file)"
                }

            # ✅ Load training dataset
            df = pd.read_csv("data/fixed_linkedin_posts1_train.csv")
            df.columns = df.columns.str.strip().str.lower()

            # Handle missing values
            df["cta_used"] = df["cta_used"].fillna("no_cta")
            df.fillna(0, inplace=True)

            # ✅ Target variable
            df["engagement_score"] = (
                df["like_count"] +
                2 * df["comment_count"] +
                3 * df["repost_count"]
            ) / df["follower_count"].replace(0, 1)

            # -------------------------
            # Encoding
            # -------------------------
            self.encoders["post_type"] = LabelEncoder()
            df["post_type_encoded"] = self.encoders["post_type"].fit_transform(df["post_type"])

            self.encoders["CTA_used"] = LabelEncoder()
            df["cta_encoded"] = self.encoders["CTA_used"].fit_transform(df["cta_used"])

            # -------------------------
            # Features (NO LEAKAGE)
            # -------------------------
            self.feature_columns = [
                "follower_count",
                "post_type_encoded",
                "hashtag_count",
                "mention_count",
                "cta_encoded"
            ]

            X = df[self.feature_columns]
            y = df["engagement_score"]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )

            self.model.fit(X_train, y_train)

            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            self.is_trained = True
            self.save_model()

            return {
                "success": True,
                "message": "Model trained from train.csv",
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4),
                "data_count": len(df)
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    # =========================
    # PREDICT
    # =========================
    def predict(
        self,
        follower_count: int,
        post_type: str,
        hashtag_count: int,
        mention_count: int,
        cta_used: str
    ) -> Dict[str, Any]:

        if not self.is_trained or self.model is None:
            return {"success": False, "message": "Model not trained yet"}

        try:
            # Normalize input
            post_type = post_type.lower()
            cta_used = cta_used.lower()

            # ✅ Safe fallback for unseen values
            if post_type not in self.encoders["post_type"].classes_:
                post_type = self.encoders["post_type"].classes_[0]

            if cta_used not in self.encoders["CTA_used"].classes_:
                cta_used = self.encoders["CTA_used"].classes_[0]

            # Encode
            post_type_encoded = self.encoders["post_type"].transform([post_type])[0]
            cta_encoded = self.encoders["CTA_used"].transform([cta_used])[0]

            # Prepare input
            features = pd.DataFrame([{
                "follower_count": follower_count,
                "post_type_encoded": post_type_encoded,
                "hashtag_count": hashtag_count,
                "mention_count": mention_count,
                "cta_encoded": cta_encoded
            }])

            features = features[self.feature_columns]

            # Prediction
            prediction = self.model.predict(features)[0]

            # -------------------------
            # Confidence Score
            # -------------------------
            preds = np.array([
                tree.predict(features.to_numpy())[0]
                for tree in self.model.estimators_
            ])

            std_dev = np.std(preds)
            mean_pred = np.mean(preds)

            confidence = max(0, 1 - (std_dev / (abs(mean_pred) + 1e-6)))

            return {
                "success": True,
                "predicted_engagement_score": round(float(prediction), 4) *100,
                "confidence_score": round(min(prediction * 10, 1), 2) 
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    # =========================
    # SAVE / LOAD
    # =========================
    def save_model(self):
        if self.model is not None:
            joblib.dump({
                "model": self.model,
                "encoders": self.encoders,
                "feature_columns": self.feature_columns,
                "is_trained": self.is_trained
            }, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data["model"]
                self.encoders = data["encoders"]
                self.feature_columns = data.get("feature_columns", [])
                self.is_trained = data["is_trained"]
            except Exception as e:
                print("Model load failed:", e)
                self.is_trained = False