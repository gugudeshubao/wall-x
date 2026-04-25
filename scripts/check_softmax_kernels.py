"""Check if cunn_SoftMaxForward appears in both FA2 and SDPA nsys traces."""
import sqlite3, sys, os

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)

# Get softmax kernel info
query = """
SELECT s.value as kernel_name, 
       COUNT(*) as count,
       SUM(k.end - k.start) / 1e6 as total_ms,
       AVG(k.end - k.start) / 1e3 as avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%softmax%' OR s.value LIKE '%Softmax%' OR s.value LIKE '%SoftMax%'
GROUP BY s.value
ORDER BY total_ms DESC
"""

try:
    rows = conn.execute(query).fetchall()
    if rows:
        print(f"{'Kernel':<80s} {'Count':>6s} {'Total(ms)':>10s} {'Avg(us)':>10s}")
        print("-" * 110)
        for name, cnt, total_ms, avg_us in rows:
            print(f"{name[:80]:<80s} {cnt:>6d} {total_ms:>10.1f} {avg_us:>10.1f}")
    else:
        print("No softmax kernels found!")
except Exception as e:
    print(f"Error: {e}")
    # Try without JOIN
    try:
        rows = conn.execute("""
            SELECT demangledName, COUNT(*), SUM(end-start)/1e6, AVG(end-start)/1e3
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE demangledName LIKE '%oftmax%' OR CAST(demangledName AS TEXT) LIKE '%oftmax%'
            GROUP BY demangledName
        """).fetchall()
        print(f"Direct query: {rows}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")

conn.close()
