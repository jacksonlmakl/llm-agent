import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect('db.duckdb')

con.execute("""
CREATE TABLE IF NOT EXISTS session (
    session_id STRING PRIMARY KEY,
    created_date TIMESTAMP DEFAULT current_localtimestamp()
)
""")
con.execute("""
CREATE TABLE IF NOT EXISTS messages (
    message_id STRING PRIMARY KEY,
    session_id STRING,
    role STRING,
    content STRING,
    created_date TIMESTAMP DEFAULT current_localtimestamp()
)
""")
con.commit()
con.close
