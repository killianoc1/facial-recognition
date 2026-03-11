import sqlite3

DB_NAME = "faces.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Drop the old table (deletes all old data — which is fine since it's incompatible)
cursor.execute("DROP TABLE IF EXISTS users")

# Create the new table with the correct columns
cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embeddings TEXT NOT NULL
    )
""")

conn.commit()
conn.close()

print("Database reset successfully! Old data cleared, new table created.")
print("Now run 'python3 main.py' and register faces again.")