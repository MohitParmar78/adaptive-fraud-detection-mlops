import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)  # 🔥 THIS IS CRITICAL

DB_PATH = os.path.join(DATA_DIR, "fraud.db")


def init_db():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Time REAL,
        V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL,
        V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL,
        V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL,
        V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL,
        V25 REAL, V26 REAL, V27 REAL, V28 REAL,
        Amount REAL,
        prediction INTEGER,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()


# 🔹 Create table
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Time REAL,
        V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL,
        V7 REAL, V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL,
        V13 REAL, V14 REAL, V15 REAL, V16 REAL, V17 REAL, V18 REAL,
        V19 REAL, V20 REAL, V21 REAL, V22 REAL, V23 REAL, V24 REAL,
        V25 REAL, V26 REAL, V27 REAL, V28 REAL,
        Amount REAL,
        prediction INTEGER,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()


# 🔹 Save data
def save_to_db(data: dict, pred: int, prob: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    values = list(data.values()) + [pred, prob]

    cursor.execute("""
    INSERT INTO transactions (
        Time, V1, V2, V3, V4, V5, V6,
        V7, V8, V9, V10, V11, V12,
        V13, V14, V15, V16, V17, V18,
        V19, V20, V21, V22, V23, V24,
        V25, V26, V27, V28, Amount,
        prediction, probability
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, values)

    conn.commit()
    conn.close()


# 🔹 Load DB data
def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("SELECT * FROM transactions", conn)

    conn.close()

    return df