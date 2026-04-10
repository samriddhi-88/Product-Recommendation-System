from user_manager import UserManager

um = UserManager()

user_id = um.register_user("Samridhi", "electronics", 2000)

print("Generated User ID:", user_id)