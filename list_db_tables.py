import duckdb

def list_tables():
    db_path = r'C:\Users\Gaming PC\Desktop\TTBall_5\ai-service\results\ttball_new.duckdb'
    try:
        con = duckdb.connect(database=db_path, read_only=True)
        tables = con.execute("SHOW TABLES").fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table[0])
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()

if __name__ == "__main__":
    list_tables()

