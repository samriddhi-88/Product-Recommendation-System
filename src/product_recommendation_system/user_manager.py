import pandas as pd
import os

class UserManager:
    def __init__(self, file_path="new_user_data/users.csv"):
        self.file_path = file_path

        # अगर file exist नहीं करती तो create करो
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=["user_id", "name", "category", "budget"])
            df.to_csv(self.file_path, index=False)

    # 🔹 Unique ID generate
    def generate_user_id(self):
        df = pd.read_csv(self.file_path)

        if df.empty:
            return "U1001"

        last_id = df['user_id'].iloc[-1]
        new_id = int(last_id[1:]) + 1
        return "U" + str(new_id)

    # 🔹 Register user
    def register_user(self, name, category, budget):
        user_id = self.generate_user_id()

        new_user = {
            "user_id": user_id,
            "name": name,
            "category": category,
            "budget": budget
        }

        df = pd.read_csv(self.file_path)
        df = pd.concat([df, pd.DataFrame([new_user])])
        df.to_csv(self.file_path, index=False)

        return user_id

    # 🔹 Check user exists
    def user_exists(self, user_id):
        df = pd.read_csv(self.file_path)
        return user_id in df['user_id'].values