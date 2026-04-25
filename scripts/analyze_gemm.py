"""Analyze GEMM kernels from nsys SQLite trace."""
import sqlite3, sys

db_path = sys.argv[1] if len(sys.argv) > 1 else '/data/wy/wall-x/profiling/fa2_nsys.sqlite'
conn = sqlite3.connect(db_path)

# Get all GEMM-like kernels with full names
rows = conn.execute('''
SELECT s.value as name, 
       COUNT(*) as cnt,
       SUM(k.end - k.start) / 1e6 as total_ms,
       AVG(k.end - k.start) / 1e3 as avg_us,
       MIN(k.end - k.start) / 1e3 as min_us,
       MAX(k.end - k.start) / 1e3 as max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%gemm%' OR s.value LIKE '%gemv%' 
   OR s.value LIKE '%cutlass%' OR s.value LIKE '%cublas%' 
   OR s.value LIKE '%Gemm%' OR s.value LIKE '%GEMM%'
GROUP BY s.value
ORDER BY total_ms DESC
''').fetchall()

print("Kernel                                                                          Cnt  Total(ms)   Avg(us)   Min(us)   Max(us)  Library")
print("-" * 140)
total_gemm_ms = 0
for name, cnt, total_ms, avg_us, min_us, max_us in rows:
    total_gemm_ms += total_ms
    lib = 'unknown'
    if 'cutlass' in name.lower():
        lib = 'CUTLASS'
    elif 'ampere_' in name:
        lib = 'cuBLAS'
    short = name[:75]
    print(f"{short:<75s} {cnt:>5d} {total_ms:>10.1f} {avg_us:>10.1f} {min_us:>10.1f} {max_us:>10.1f}  {lib}")

print(f"\nTotal GEMM time: {total_gemm_ms:.1f} ms")

total_gpu = conn.execute('SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
print(f"Total GPU kernel time: {total_gpu:.1f} ms")
print(f"GEMM fraction: {total_gemm_ms/total_gpu*100:.1f}%")

# Breakdown by library
print("\n=== Library breakdown ===")
lib_totals = {}
for name, cnt, total_ms, avg_us, min_us, max_us in rows:
    lib = 'CUTLASS' if 'cutlass' in name.lower() else ('cuBLAS' if 'ampere_' in name else 'unknown')
    if lib not in lib_totals:
        lib_totals[lib] = [0, 0]
    lib_totals[lib][0] += cnt
    lib_totals[lib][1] += total_ms

for lib, (cnt, ms) in sorted(lib_totals.items(), key=lambda x: -x[1][1]):
    print(f"  {lib:<10s}: {cnt:>6d} calls, {ms:>8.1f} ms ({ms/total_gemm_ms*100:.1f}%)")

# Decode vs Prefill analysis: group by grid size
print("\n=== Grid size distribution (proxy for decode vs prefill) ===")
grid_rows = conn.execute('''
SELECT k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
       s.value as name,
       COUNT(*) as cnt,
       SUM(k.end - k.start) / 1e6 as total_ms,
       AVG(k.end - k.start) / 1e3 as avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE (s.value LIKE '%gemm%' OR s.value LIKE '%gemv%' 
   OR s.value LIKE '%cutlass%' OR s.value LIKE '%Gemm%')
GROUP BY k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, s.value
ORDER BY total_ms DESC
LIMIT 15
''').fetchall()

print(f"{'Grid':<20s} {'Block':<15s} {'Kernel':<55s} {'Cnt':>5s} {'Total(ms)':>10s} {'Avg(us)':>10s}")
print("-" * 120)
for gx, gy, gz, bx, by, bz, name, cnt, total_ms, avg_us in grid_rows:
    grid_str = f"({gx},{gy},{gz})"
    block_str = f"({bx},{by},{bz})"
    print(f"{grid_str:<20s} {block_str:<15s} {name[:55]:<55s} {cnt:>5d} {total_ms:>10.1f} {avg_us:>10.1f}")

conn.close()
