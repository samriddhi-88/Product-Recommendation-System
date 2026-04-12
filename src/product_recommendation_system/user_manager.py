import pandas as pd
import numpy as np
import uuid
import os
import hashlib
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
 
USERS_COLS = [
    "user_id", "name", "email", "password_hash",
    "category_pref", "price_pref",
    "segment", "total_interactions", "joined_date",
    "customer_unique_id"
]
INTERACTION_COLS = [
    "interaction_id", "user_id", "product_id",
    "category", "price", "action_type", "timestamp"
]
 
PRICE_LABELS = ["budget", "low", "mid", "mid-high", "high", "premium"]
 
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
 
 
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.strip().encode()).hexdigest()
 
 
class UserManager:
 
    def __init__(self):
        self._gc     = None
        self._sheet  = None
        self._ws_users = None
        self._ws_inter = None
        self._connect()
 
    def _connect(self):
        try:
            import streamlit as st
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            self._gc    = gspread.authorize(creds)
            sheet_id    = st.secrets["SHEET_ID"]
            self._sheet = self._gc.open_by_key(sheet_id)
 
            # Users worksheet
            try:
                self._ws_users = self._sheet.worksheet("Users")
            except gspread.WorksheetNotFound:
                self._ws_users = self._sheet.add_worksheet("Users", rows=1000, cols=20)
                self._ws_users.append_row(USERS_COLS)
 
            # Interactions worksheet
            try:
                self._ws_inter = self._sheet.worksheet("Interactions")
            except gspread.WorksheetNotFound:
                self._ws_inter = self._sheet.add_worksheet("Interactions", rows=5000, cols=20)
                self._ws_inter.append_row(INTERACTION_COLS)
 
            # Add headers if sheet is empty
            if not self._ws_users.get_all_values():
                self._ws_users.append_row(USERS_COLS)
            if not self._ws_inter.get_all_values():
                self._ws_inter.append_row(INTERACTION_COLS)
 
        except Exception as e:
            print(f"Google Sheets connection error: {e}")
 
    def _load_users(self):
        try:
            data = self._ws_users.get_all_records()
            if not data:
                return pd.DataFrame(columns=USERS_COLS)
            df = pd.DataFrame(data)
            for col in USERS_COLS:
                if col not in df.columns:
                    df[col] = ""
            df["total_interactions"] = pd.to_numeric(
                df["total_interactions"], errors="coerce"
            ).fillna(0).astype(int)
            return df
        except Exception as e:
            print(f"Error loading users: {e}")
            return pd.DataFrame(columns=USERS_COLS)
 
    def _save_user_row(self, row_dict):
        """Append a new user row to Google Sheet."""
        try:
            row = [str(row_dict.get(col, "")) for col in USERS_COLS]
            self._ws_users.append_row(row)
        except Exception as e:
            print(f"Error saving user: {e}")
 
    def _update_user_row(self, user_id, updates: dict):
        """Update specific fields for a user in Google Sheet."""
        try:
            data = self._ws_users.get_all_values()
            headers = data[0]
            for i, row in enumerate(data[1:], start=2):
                if row[headers.index("user_id")] == user_id:
                    for key, val in updates.items():
                        col_idx = headers.index(key) + 1
                        self._ws_inter  # just to keep reference
                        self._ws_users.update_cell(i, col_idx, str(val))
                    break
        except Exception as e:
            print(f"Error updating user: {e}")
 
    def _load_interactions(self):
        try:
            data = self._ws_inter.get_all_records()
            if not data:
                return pd.DataFrame(columns=INTERACTION_COLS)
            df = pd.DataFrame(data)
            for col in INTERACTION_COLS:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception as e:
            print(f"Error loading interactions: {e}")
            return pd.DataFrame(columns=INTERACTION_COLS)
 
    def _save_interaction_row(self, row_dict):
        """Append a new interaction row to Google Sheet."""
        try:
            row = [str(row_dict.get(col, "")) for col in INTERACTION_COLS]
            self._ws_inter.append_row(row)
        except Exception as e:
            print(f"Error saving interaction: {e}")
 
    def _get_segment(self, total):
        if total == 0:   return "A"
        elif total == 1: return "B"
        elif total <= 4: return "C"
        else:            return "D"
 
    # ── Register New User ──────────────────────────────────────────────────
    def register_user(self, name, email, password, category_prefs, price_pref):
        df = self._load_users()
        if not df.empty and email.strip().lower() in df["email"].str.lower().values:
            return {"status": "exists", "user_id": None,
                    "message": "Email already registered. Please login."}
 
        user_id = "USR-" + str(uuid.uuid4())[:8].upper()
        new_row = {
            "user_id"            : user_id,
            "name"               : name.strip(),
            "email"              : email.strip().lower(),
            "password_hash"      : _hash_password(password),
            "category_pref"      : ",".join(category_prefs),
            "price_pref"         : price_pref,
            "segment"            : "A",
            "total_interactions" : 0,
            "joined_date"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_unique_id" : ""
        }
        self._save_user_row(new_row)
        return {"status": "success", "user_id": user_id,
                "message": f"Welcome {name}! Account ban gaya!"}
 
    # ── Register Existing Customer ─────────────────────────────────────────
    def register_existing_customer(self, customer_unique_id, name, email, password, category_prefs, price_pref, csv_path):
        try:
            df_raw = pd.read_csv(csv_path, usecols=["customer_unique_id", "order_status"])
            customer_rows = df_raw[df_raw["customer_unique_id"] == customer_unique_id.strip()]
 
            if len(customer_rows) == 0:
                return {"status": "not_found", "user_id": None,
                        "message": "Yeh Customer ID dataset mein nahi mili. Check karo."}
 
            actual_interactions = len(customer_rows[customer_rows["order_status"] == "delivered"])
            actual_segment      = self._get_segment(actual_interactions)
 
        except Exception as e:
            return {"status": "error", "user_id": None,
                    "message": f"Dataset verify nahi ho saka: {str(e)}"}
 
        df = self._load_users()
        if not df.empty:
            if email.strip().lower() in df["email"].str.lower().values:
                return {"status": "exists", "user_id": None,
                        "message": "Email already registered. Please login."}
            if customer_unique_id.strip() in df["customer_unique_id"].values:
                return {"status": "exists", "user_id": None,
                        "message": "Yeh Customer ID already registered hai. Please login."}
 
        user_id = "USR-" + str(uuid.uuid4())[:8].upper()
        new_row = {
            "user_id"            : user_id,
            "name"               : name.strip(),
            "email"              : email.strip().lower(),
            "password_hash"      : _hash_password(password),
            "category_pref"      : ",".join(category_prefs),
            "price_pref"         : price_pref,
            "segment"            : actual_segment,
            "total_interactions" : actual_interactions,
            "joined_date"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_unique_id" : customer_unique_id.strip()
        }
        self._save_user_row(new_row)
 
        return {
            "status"       : "success",
            "user_id"      : user_id,
            "segment"      : actual_segment,
            "interactions" : actual_interactions,
            "message"      : f"Welcome back {name}! Aapka purana data mil gaya! 🎉"
        }
 
    # ── Login ──────────────────────────────────────────────────────────────
    def login_user(self, email, password):
        df = self._load_users()
        email = email.strip().lower()
        match = df[df["email"] == email]
        if match.empty:
            return {"status": "not_found", "user": None,
                    "message": "Email nahi mila. Pehle register karo."}
        user = match.iloc[0].to_dict()
        stored_hash = user.get("password_hash", "")
        if stored_hash != _hash_password(password):
            return {"status": "wrong_password", "user": None,
                    "message": "Password galat hai. Dobara try karo."}
        return {"status": "success", "user": user,
                "message": f"Welcome back, {user['name']}!"}
 
    # ── Get User ───────────────────────────────────────────────────────────
    def get_user(self, user_id):
        df = self._load_users()
        match = df[df["user_id"] == user_id.strip().upper()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()
 
    # ── Get customer_unique_id for NCF ────────────────────────────────────
    def get_customer_unique_id(self, user_id):
        user = self.get_user(user_id)
        if user is None:
            return None
        cuid = user.get("customer_unique_id", "")
        if not cuid or str(cuid) == "nan" or cuid == "":
            return None
        return cuid
 
    # ── Log Interaction ────────────────────────────────────────────────────
    def log_interaction(self, user_id, product_id, category, price, action_type="click"):
        new_row = {
            "interaction_id" : "INT-" + str(uuid.uuid4())[:8].upper(),
            "user_id"        : user_id,
            "product_id"     : product_id,
            "category"       : category,
            "price"          : round(float(price), 2) if price else 0.0,
            "action_type"    : action_type,
            "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_interaction_row(new_row)
 
        # Update user segment
        df = self._load_users()
        idx = df[df["user_id"] == user_id].index
        if not idx.empty:
            i     = idx[0]
            total = int(df.at[i, "total_interactions"]) + 1
            seg   = self._get_segment(total)
            self._update_user_row(user_id, {
                "total_interactions": total,
                "segment": seg
            })
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
        self._update_user_row(user_id, {
            "category_pref": ",".join(category_prefs),
            "price_pref"   : price_pref
        })
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