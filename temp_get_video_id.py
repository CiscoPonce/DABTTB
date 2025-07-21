import duckdb
conn = duckdb.connect('/app/results/ttball_new.duckdb')
result = conn.execute("SELECT id FROM video_metadata WHERE filename = 'Video-anomalies2.mp4'").fetchone()
if result:
    print(result[0])
else:
    print("None")
conn.close()