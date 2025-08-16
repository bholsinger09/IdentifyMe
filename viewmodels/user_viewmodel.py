from models.user import User

class UserViewModel:
    def __init__(self, user: User):
        self.user = user

    def display_name(self):
        return f"{self.user.username} <{self.user.email}>"

    def activate(self):
        self.user.is_active = True

    def deactivate(self):
        self.user.is_active = False
