import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# --- NEW: Cloud Database Connection via SQLAlchemy ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DB_PATH = os.path.join(BASE_DIR, "data", "fraud.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"

# Initialize Engine and Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 🔹 Define the schema for the transactions table
class TransactionLog(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    Time = Column(Float)
    V1 = Column(Float); V2 = Column(Float); V3 = Column(Float); V4 = Column(Float)
    V5 = Column(Float); V6 = Column(Float); V7 = Column(Float); V8 = Column(Float)
    V9 = Column(Float); V10 = Column(Float); V11 = Column(Float); V12 = Column(Float)
    V13 = Column(Float); V14 = Column(Float); V15 = Column(Float); V16 = Column(Float)
    V17 = Column(Float); V18 = Column(Float); V19 = Column(Float); V20 = Column(Float)
    V21 = Column(Float); V22 = Column(Float); V23 = Column(Float); V24 = Column(Float)
    V25 = Column(Float); V26 = Column(Float); V27 = Column(Float); V28 = Column(Float)
    Amount = Column(Float)
    prediction = Column(Integer)
    probability = Column(Float)
    
    # 🔥 HUMAN-IN-THE-LOOP UPGRADE: The true label provided by a human analyst later
    Actual_Class = Column(Integer, nullable=True)

def init_db():
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)

def save_to_db(data: dict, pred: int, prob: float):
    db = SessionLocal()
    try:
        new_tx = TransactionLog(**data, prediction=pred, probability=prob)
        db.add(new_tx)
        db.commit()
    except Exception as e:
        # 🔥 THE FIX: If anything goes wrong, roll back the transaction so the DB doesn't freeze
        db.rollback()
        print(f"⚠️ Database insertion failed: {e}")
    finally:
        db.close()

def load_data_from_db():
    """Load all database records into a Pandas DataFrame for Retraining/Drift detection."""
    return pd.read_sql("SELECT * FROM transactions", engine)