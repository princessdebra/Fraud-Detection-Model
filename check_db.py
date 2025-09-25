
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# explicitly tell dotenv where to load
load_dotenv("db.env")

db_url = os.getenv("DB_URL")
print("DB_URL:", db_url)

db_url = os.getenv("DB_URL")
print("DB_URL:", db_url)

engine = create_engine(db_url, pool_pre_ping=True, future=True)

with engine.connect() as c:
    print("Database():", c.execute(text("SELECT DATABASE()")).scalar_one())
    print(
        "Users table exists?:",
        bool(
            c.execute(
                text("""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    AND table_name = 'users'
                """)
            ).scalar()
        ),
    )
