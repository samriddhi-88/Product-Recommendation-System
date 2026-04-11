import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime
 
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
NEW_USER_DIR  = os.path.join(PROJECT_ROOT, "new_user_data")
USERS_CSV     = os.path.join(NEW_USER_DIR, "users.csv")
INTERACTION_CSV = os.path.join(NEW_USER_DIR, "interaction.csv")
 
USERS_COLS = [
    "user_id", "name", "email",
    "category_pref", "price_pref",
    "segment", "total_interactions", "joined_date"
]
INTERACTION_COLS = [
    "interaction_id", "user_id", "product_id",
    "category", "price", "action_type", "timestamp"
]
 
PRICE_LABELS = ["budget", "low", "mid", "mid-high", "high", "premium"]
 
 
class UserManager:
 
    def __init__(self):
        os.makedirs(NEW_USER_DIR, exist_ok=True)
        self._init_csv(USERS_CSV, USERS_COLS)
        self._init_csv(INTERACTION_CSV, INTERACTION_COLS)
 
    def _init_csv(self, path, columns):
        if not os.path.exists(path):
            pd.DataFrame(columns=columns).to_csv(path, index=False)
 
    def _load_users(self):
        df = pd.read_csv(USERS_CSV, dtype=str)
        if df.empty:
            return pd.DataFrame(columns=USERS_COLS)
        df["total_interactions"] = pd.to_numeric(
            df["total_interactions"], errors="coerce"
        ).fillna(0).astype(int)
        return df
 
    def _save_users(self, df):
        df.to_csv(USERS_CSV, index=False)
 
    def _load_interactions(self):
        df = pd.read_csv(INTERACTION_CSV, dtype=str)
        if df.empty:
            return pd.DataFrame(columns=INTERACTION_COLS)
        return df
 
    def _save_interactions(self, df):
        df.to_csv(INTERACTION_CSV, index=False)
 
    def _get_segment(self, total):
        if total == 0:   return "A"
        elif total == 1: return "B"
        elif total <= 4: return "C"
        else:            return "D"
 
    def register_user(self, name, email, category_prefs, price_pref):
        df = self._load_users()
        if not df.empty and email.strip().lower() in df["email"].str.lower().values:
            return {"status": "exists", "user_id": None,
                    "message": "Email already registered. Please login."}
 
        user_id = "USR-" + str(uuid.uuid4())[:8].upper()
        new_row = {
            "user_id"            : user_id,
            "name"               : name.strip(),
            "email"              : email.strip().lower(),
            "category_pref"      : ",".join(category_prefs),
            "price_pref"         : price_pref,
            "segment"            : "A",
            "total_interactions" : 0,
            "joined_date"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_users(df)
        return {"status": "success", "user_id": user_id,
                "message": f"Welcome {name}! Tumhara ID: {user_id}"}
 
    def login_user(self, user_id):
        df = self._load_users()
        match = df[df["user_id"] == user_id.strip().upper()]
        if match.empty:
            return {"status": "not_found", "user": None,
                    "message": "User ID nahi mila. Check karo ya register karo."}
        user = match.iloc[0].to_dict()
        return {"status": "success", "user": user,
                "message": f"Welcome back, {user['name']}!"}
 
    def get_user(self, user_id):
        df = self._load_users()
        match = df[df["user_id"] == user_id.strip().upper()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()
 
    def log_interaction(self, user_id, product_id, category, price, action_type="click"):
        interactions = self._load_interactions()
        new_row = {
            "interaction_id" : "INT-" + str(uuid.uuid4())[:8].upper(),
            "user_id"        : user_id,
            "product_id"     : product_id,
            "category"       : category,
            "price"          : round(float(price), 2) if price else 0.0,
            "action_type"    : action_type,
            "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        interactions = pd.concat(
            [interactions, pd.DataFrame([new_row])], ignore_index=True
        )
        self._save_interactions(interactions)
 
        users = self._load_users()
        idx   = users[users["user_id"] == user_id].index
        if not idx.empty:
            i = idx[0]
            total = int(users.at[i, "total_interactions"]) + 1
            users.at[i, "total_interactions"] = total
            users.at[i, "segment"]            = self._get_segment(total)
            self._save_users(users)
        return True
 
    def get_user_interactions(self, user_id):
        df = self._load_interactions()
        if df.empty:
            return pd.DataFrame(columns=INTERACTION_COLS)
        result = df[df["user_id"] == user_id].copy()
        if not result.empty:
            result = result.sort_values("timestamp", ascending=False)
        return result.reset_index(drop=True)
 
    def get_user_segment(self, user_id):
        user = self.get_user(user_id)
        if user is None: return "A"
        return user.get("segment", "A")
 
    def get_user_category_prefs(self, user_id):
        user = self.get_user(user_id)
        if user is None: return []
        prefs = user.get("category_pref", "")
        if not prefs or str(prefs) == "nan": return []
        return [c.strip() for c in str(prefs).split(",") if c.strip()]
 
    def get_user_price_pref(self, user_id):
        user = self.get_user(user_id)
        if user is None: return "mid"
        return user.get("price_pref", "mid")
 
    def update_user_prefs(self, user_id, category_prefs, price_pref):
        users = self._load_users()
        idx   = users[users["user_id"] == user_id].index
        if idx.empty: return False
        i = idx[0]
        users.at[i, "category_pref"] = ",".join(category_prefs)
        users.at[i, "price_pref"]    = price_pref
        self._save_users(users)
        return True
 
    def get_all_users(self):
        return self._load_users()
 
    def get_all_interactions(self):
        return self._load_interactions()
 
    def get_stats(self):
        users        = self._load_users()
        interactions = self._load_interactions()
        seg_counts   = {"A": 0, "B": 0, "C": 0, "D": 0}
        if not users.empty:
            for seg, cnt in users["segment"].value_counts().items():
                if seg in seg_counts:
                    seg_counts[seg] = int(cnt)
        return {
            "total_users"        : len(users),
            "total_interactions" : len(interactions),
            "segment_counts"     : seg_counts,
        }
    