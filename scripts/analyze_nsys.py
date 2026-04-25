#!/usr/bin/env python3
"""Analyze nsys sqlite for top kernels."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "profiling/fa2_nsys.sqlite"
db = sqlite3.connect(db_path)
cur = db.cursor()

# Total GPU time
cur.execute("SELECT sum(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
total_gpu_ns = cur.fetchone()[0]
total_gpu_ms = total_gpu_ns / 1e6

print(f"Total GPU kernel time: {total_gpu_ms:.1f} ms")
print(f"Total kernels: ", end="")
cur.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
print(cur.fetchone()[0])
print()

# Top 25 kernels by total time
cur.execute("""
SELECT
    s.value as kname,
    count(*) as calls,
    sum(k.end-k.start) as total_ns,
    avg(k.end-k.start) as avg_ns,
    min(k.end-k.start) as min_ns,
    max(k.end-k.start) as max_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
ORDER BY total_ns DESC
LIMIT 25
""")

header = f"{'#':>3} {'%GPU':>6} {'Total(ms)':>10} {'Calls':>6} {'Avg(us)':>10} {'Kernel'}"
print(header)
print("-" * len(header) + "-" * 80)

rows = cur.fetchall()
for i, r in enumerate(rows):
    name = str(r[0]) if r[0] else "unknown"
    if len(name) > 110:
        name = name[:107] + "..."
    pct = r[2] / total_gpu_ns * 100
    print(f"{i+1:>3} {pct:>5.1f}% {r[2]/1e6:>10.1f} {r[1]:>6} {r[3]/1e3:>10.1f}  {name}")

print()

# FA2 specific: flash_fwd kernel
print("=" * 80)
print("Flash Attention kernels:")
print("=" * 80)
cur.execute("""
SELECT
    s.value as kname,
    count(*) as calls,
    sum(k.end-k.start) as total_ns,
    avg(k.end-k.start) as avg_ns,
    min(k.end-k.start) as min_ns,
    max(k.end-k.start) as max_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%flash%'
   OR s.value LIKE '%fmha%'
   OR s.value LIKE '%Flash%'
GROUP BY s.value
ORDER BY total_ns DESC
""")
rows = cur.fetchall()
if not rows:
    print("  (no flash attention kernels found)")
else:
    for r in rows:
        name = str(r[0]) if r[0] else "unknown"
        if len(name) > 120:
            name = name[:117] + "..."
        pct = r[2] / total_gpu_ns * 100
        print(f"  {pct:>5.1f}% {r[2]/1e6:>8.1f}ms  calls={r[1]:>4}  avg={r[3]/1e3:.1f}us  min={r[4]/1e3:.1f}us  max={r[5]/1e3:.1f}us")
        print(f"         {name}")

print()

# Contiguous/copy kernels
print("=" * 80)
print("Copy/contiguous kernels:")
print("=" * 80)
cur.execute("""
SELECT
    s.value as kname,
    count(*) as calls,
    sum(k.end-k.start) as total_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%copy%'
   OR s.value LIKE '%contiguous%'
   OR s.value LIKE '%Clone%'
GROUP BY s.value
ORDER BY total_ns DESC
LIMIT 10
""")
rows = cur.fetchall()
for r in rows:
    name = r[0][:100]
    pct = r[2] / total_gpu_ns * 100
    print(f"  {pct:>5.1f}% {r[2]/1e6:>8.1f}ms  calls={r[1]:>4}  {name}")

# Kernel category summary
print()
print("=" * 80)
print("Kernel category summary:")
print("=" * 80)
categories = [
    ("GEMM (cuBLAS)", "%gemm%"),
    ("GEMM (cublas)", "%cublas%"),
    ("Flash/FMHA", "%flash%"),
    ("Softmax", "%softmax%"),
    ("LayerNorm/RMSNorm", "%norm%"),
    ("Elementwise", "%elementwise%"),
    ("Reduce", "%reduce%"),
    ("Copy/memcpy", "%copy%"),
    ("Custom ops (wallx)", "%wallx%"),
    ("RoPE (rot_pos_emb)", "%rot_pos%"),
    ("Window index", "%window%"),
]
for cat_name, pattern in categories:
    cur.execute(f"""
    SELECT count(*), COALESCE(sum(k.end-k.start),0)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE lower(s.value) LIKE '{pattern}'
    """)
    cnt, ns = cur.fetchone()
    if cnt > 0:
        pct = ns / total_gpu_ns * 100
        print(f"  {cat_name:<30} {pct:>5.1f}%  {ns/1e6:>8.1f}ms  ({cnt} calls)")

db.close()
