# database.py  (replace file)
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import text
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from passlib.hash import bcrypt
import logging
load_dotenv("db.env")

DB_URL = os.getenv("DB_URL", "sqlite:///fraud_detection.db")  # fallback
FERNET_KEY = os.getenv("FERNET_KEY")
fernet = Fernet(FERNET_KEY) if FERNET_KEY else None

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()
log = logging.getLogger(__name__)
# ---------- Models ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="analyst")  # analyst|manager|admin
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Assignment(Base):
    __tablename__ = "assignments"
    id = Column(Integer, primary_key=True)
    visit_id = Column(String(128), index=True)
    assignee = Column(String(255))
    due_date = Column(String(32))
    assigned_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="pending")  # pending|done|discarded
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    visit_id = Column(String(128), index=True)
    note_text_enc = Column(LONGTEXT)  # encrypted
    created_date = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    visit_id = Column(String(128), index=True)
    verdict = Column(String(64))
    comments_enc = Column(LONGTEXT)  # encrypted
    reviewer = Column(String(255))
    feedback_date = Column(DateTime, default=datetime.utcnow)

class Settings(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, index=True)   # "app_settings"
    payload = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    at = Column(DateTime, default=datetime.utcnow)
    user = Column(String(255))
    action = Column(String(255))
    details = Column(LONGTEXT)

Index("ix_feedback_visit_verdict", Feedback.visit_id, Feedback.verdict)

def _enc(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return fernet.encrypt(s.encode()).decode() if fernet else s

def _dec(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return fernet.decrypt(s.encode()).decode() if fernet else s

# ---------- Facade used by the app ----------
class FraudDetectionDB:
    def __init__(self):
        self._fernet: Optional[Fernet] = None
        key = os.getenv("MINET_NOTE_KEY")  # 32-byte urlsafe base64 key
        if Fernet and key:
            try:
                self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            except Exception:
                self._fernet = None
        Base.metadata.create_all(engine)

    # -------- auth ----------
    def create_user(self, email: str, name: str, password: str, role: str = "analyst") -> bool:
        with SessionLocal() as s:
            if s.query(User).filter_by(email=email).first():
                return False
            s.add(User(email=email, name=name, password_hash=bcrypt.hash(password), role=role))
            s.commit()
            return True
    def _encrypt(self, txt: str) -> str:
        if not txt:
            return ""
        if self._fernet:
            return self._fernet.encrypt(txt.encode("utf-8")).decode("utf-8")
        # no crypto available -> store as-is (or base64 if you prefer)
        return txt

    def _decrypt(self, token: str) -> str:
        if token is None:
            return ""
        if self._fernet:
            try:
                return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
            except Exception:
                return ""
        return token
    def verify_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        with SessionLocal() as s:
            u = s.query(User).filter_by(email=email, active=True).first()
            if u and bcrypt.verify(password, u.password_hash):
                return {"id": u.id, "email": u.email, "name": u.name, "role": u.role}
            return None

    # -------- audit ----------
    def audit(self, user: str, action: str, details: Dict[str, Any]):
        with SessionLocal() as s:
            s.add(AuditLog(user=user, action=action, details=json.dumps(details)))
            s.commit()

    # -------- app data ----------
    def add_assignment(self, visit_id: str, assignee: str, due_date: str, created_by_email: str = "") -> bool:
        try:
            with SessionLocal() as s:
                creator = s.query(User).filter_by(email=created_by_email).first() if created_by_email else None
                s.add(Assignment(visit_id=visit_id, assignee=assignee, due_date=due_date,
                                 created_by=(creator.id if creator else None)))
                s.commit()
            self.audit(created_by_email or "system", "add_assignment",
                       {"visit_id": visit_id, "assignee": assignee, "due": due_date})
            return True
        except SQLAlchemyError:
            return False

    def add_note(self, visit_id: str, note_text: str, created_by: str):
    # normalize to a string
        note_text = (note_text or "").strip()

    # encrypt OR store raw, but always define the variable we insert
        enc = self._encrypt(note_text) if note_text else ""   # whatever your encrypt fn is named

        with self.Session() as s:
            s.execute(
                text("""
                INSERT INTO notes (visit_id, note_text_enc, created_date, created_by)
                VALUES (:visit_id, :note_text_enc, NOW(), :created_by)
            """),
            {
                "visit_id": visit_id,
                "note_text_enc": enc,
                "created_by": created_by,
            },
        )
        s.commit()

    def list_notes(self, visit_id: str):
        with self._engine.begin() as conn:
            rows = conn.execute(
                text("""SELECT id, visit_id, note_text, created_by, created_at
                        FROM notes WHERE visit_id = :vid ORDER BY created_at DESC"""),
                {"vid": visit_id},
            ).mappings().all()
        for r in rows:
            r["note_text"] = self._decrypt(r["note_text"])
        return rows

    def add_feedback(self, visit_id: str, verdict: str, comments: str, reviewer: str) -> bool:
        try:
            with SessionLocal() as s:
                s.add(Feedback(visit_id=visit_id, verdict=verdict,
                               comments_enc=_enc(comments), reviewer=reviewer))
                s.commit()
            self.audit(reviewer or "system", "add_feedback", {"visit_id": visit_id, "verdict": verdict})
            return True
        except SQLAlchemyError:
            return False

    def get_feedback_stats(self):
        import pandas as pd
        with SessionLocal() as s:
            rows = s.query(Feedback).all()
            if not rows:
                return pd.DataFrame()
            data = [{
                "visit_id": r.visit_id,
                "verdict": r.verdict,
                "comments": _dec(r.comments_enc),
                "reviewer": r.reviewer,
                "feedback_date": r.feedback_date,
            } for r in rows]
            return pd.DataFrame(data)

    def save_settings(self, payload: Dict[str, Any]) -> bool:
        try:
            with SessionLocal() as s:
                row = s.query(Settings).filter_by(key="app_settings").first()
                if row:
                    row.payload = payload
                    row.updated_at = datetime.utcnow()
                else:
                    s.add(Settings(key="app_settings", payload=payload))
                s.commit()
            self.audit("system", "save_settings", {})
            return True
        except SQLAlchemyError:
            return False
    # Pseudocode signatures
    def create_password_reset(self, email: str, token: str, expires_at: datetime) -> None: ...
    def lookup_password_reset(self, token: str) -> dict | None: ...
    def consume_password_reset(self, token: str) -> None: ...
    def update_user_password(self, email: str, new_password: str) -> None: ...

