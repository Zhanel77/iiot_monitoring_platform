from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.user import User


def create_default_admin(db: Session):
    existing = (
        db.query(User)
        .filter(User.email == "admin@example.com")
        .first()
    )

    if existing:
        return

    admin = User(
        email="admin@example.com",
        full_name="Default Admin",
        password_hash=hash_password("123123123"),
        role="admin",
        is_active=True,
    )

    db.add(admin)
    db.commit()

    print("Default admin created")