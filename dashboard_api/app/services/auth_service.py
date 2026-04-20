from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User
from app.repositories.user_repository import UserRepository


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def register(self, *, email: str, full_name: str, password: str, role: str) -> User:
        existing = self.user_repo.get_by_email(email)
        if existing:
            raise ValueError("User with this email already exists")

        return self.user_repo.create(
            email=email,
            full_name=full_name,
            password_hash=hash_password(password),
            role=role if role else "viewer",
        )

    def login(self, *, email: str, password: str) -> tuple[str, User]:
        user = self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("Invalid credentials")

        if not user.is_active:
            raise ValueError("User is inactive")

        if not verify_password(password, user.password_hash):
            raise ValueError("Invalid credentials")

        token = create_access_token(subject=str(user.id), role=user.role)
        return token, user