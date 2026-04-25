#!/usr/bin/env python3
"""Parse ncu CSV output for FA2 kernel metrics."""
import sys
import csv
import io

# Read ncu --csv --page raw output from stdin or file
raw = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()

# The CSV has a header row and data rows
# Find lines that look like CSV (start with " or contain the header)
lines = raw.strip().split('\n')
header_line = None
data_lines = []
for i, line in enumerate(lines):
    if '"ID",' in line or line.startswith('"ID"'):
        header_line = i
    elif header_line is not None and line.strip():
        data_lines.append(line)

if header_line is None:
    print("ERROR: Could not find CSV header")
    sys.exit(1)

header = lines[header_line]
reader = csv.reader(io.StringIO(header + '\n' + '\n'.join(data_lines)))
rows = list(reader)
headers = rows[0]

# Find column indices for key metrics
def find_col(name):
    for i, h in enumerate(headers):
        if name in h:
            return i
    return None

col_kernel = find_col('launch__kernel_name')
col_grid_x = find_col('launch__grid_dim_x')
col_grid_y = find_col('launch__grid_dim_y')
col_grid_z = find_col('launch__grid_dim_z')
col_block = find_col('launch__block_size')
col_duration = find_col('gpu__time_duration.sum')
col_l2_read_hit = find_col('lts__t_sector_op_read_hit_rate.pct')
col_l2_write_hit = find_col('lts__t_sector_op_write_hit_rate.pct')
col_dram_throughput = find_col('dram__throughput.avg.pct_of_peak')
col_sm_throughput = find_col('sm__throughput.avg.pct_of_peak')
col_occupancy = find_col('sm__warps_active.avg.pct_of_peak')
col_mem_throughput = find_col('gpu__compute_memory_throughput.avg.pct')
col_smem = find_col('launch__shared_mem_per_block_dynamic')
col_regs = find_col('launch__registers_per_thread"')
if col_regs is None:
    col_regs = find_col('launch__registers_per_thread_allocated')

def clean_num(s):
    if s is None or s == '':
        return 0.0
    s = s.replace(',', '').strip()
    # Remove unit suffixes like 'us', 'ns', 'ms', '%', 'KB', etc
    for suffix in ['us', 'ns', 'ms', 'KB', 'MB', 'GB', '%']:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break
    try:
        return float(s)
    except ValueError:
        return 0.0

print("=" * 100)
print("FA2 Kernel NCU Profiling Results (Orin, SM 8.7, GPU 1300.5 MHz)")
print("=" * 100)

for i, row in enumerate(rows[1:], 1):
    if col_kernel is None or col_kernel >= len(row):
        continue
    kname = row[col_kernel]
    # Extract short kernel name
    if 'flash_fwd_kernel<' in kname:
        short = 'flash_fwd_kernel'
        # Extract template params  
        if 'splitkv_combine' in kname:
            short = 'flash_fwd_splitkv_combine_kernel'
        elif 'splitkv' in kname:
            short = 'flash_fwd_splitkv_kernel'
    elif 'flash_fwd_splitkv_combine' in kname:
        short = 'flash_fwd_splitkv_combine_kernel'
    elif 'flash_fwd_splitkv' in kname:
        short = 'flash_fwd_splitkv_kernel'
    elif 'flash_fwd' in kname:
        short = 'flash_fwd_kernel'
    else:
        short = kname[:60]

    # Extract head_dim from template if possible
    traits = ""
    if 'Flash_fwd_kernel_traits<' in kname:
        start = kname.index('Flash_fwd_kernel_traits<') + len('Flash_fwd_kernel_traits<')
        end = kname.index('>', start)
        params = kname[start:end].split(',')
        if len(params) >= 3:
            traits = f"head_dim={params[0].strip()}, Br={params[1].strip()}, Bd={params[2].strip()}"

    grid = f"({row[col_grid_x] if col_grid_x else '?'}, {row[col_grid_y] if col_grid_y else '?'}, {row[col_grid_z] if col_grid_z else '?'})"
    
    duration = clean_num(row[col_duration]) if col_duration and col_duration < len(row) else 0
    l2_read_hit = clean_num(row[col_l2_read_hit]) if col_l2_read_hit and col_l2_read_hit < len(row) else 0
    l2_write_hit = clean_num(row[col_l2_write_hit]) if col_l2_write_hit and col_l2_write_hit < len(row) else 0
    dram_tp = clean_num(row[col_dram_throughput]) if col_dram_throughput and col_dram_throughput < len(row) else 0
    sm_tp = clean_num(row[col_sm_throughput]) if col_sm_throughput and col_sm_throughput < len(row) else 0
    occ = clean_num(row[col_occupancy]) if col_occupancy and col_occupancy < len(row) else 0
    mem_tp = clean_num(row[col_mem_throughput]) if col_mem_throughput and col_mem_throughput < len(row) else 0
    smem = clean_num(row[col_smem]) if col_smem and col_smem < len(row) else 0
    regs = clean_num(row[col_regs]) if col_regs and col_regs < len(row) else 0
    block_size = clean_num(row[col_block]) if col_block and col_block < len(row) else 0

    print(f"\n--- Kernel {i}: {short} ---")
    if traits:
        print(f"  Template:     {traits}")
    print(f"  Grid:         {grid}")
    print(f"  Block size:   {int(block_size)}")
    print(f"  Registers/thread: {int(regs)}")
    print(f"  Shared mem:   {smem/1024:.1f} KB (dynamic)")
    print(f"  Duration:     {duration:.1f} us")
    print(f"  L2 read hit:  {l2_read_hit:.1f}%")
    print(f"  L2 write hit: {l2_write_hit:.1f}%")
    print(f"  DRAM throughput:  {dram_tp:.1f}% of peak")
    print(f"  SM throughput:    {sm_tp:.1f}% of peak")
    print(f"  Compute mem TP:   {mem_tp:.1f}% of peak")
    print(f"  Occupancy:        {occ:.1f}% of peak")

# Summary stats
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

# Collect by kernel type
from collections import defaultdict
by_type = defaultdict(lambda: {'count': 0, 'l2_read': [], 'l2_write': [], 'dram': [], 'sm': [], 'occ': [], 'dur': []})

for row in rows[1:]:
    if col_kernel is None or col_kernel >= len(row):
        continue
    kname = row[col_kernel]
    if 'splitkv_combine' in kname:
        t = 'splitkv_combine'
    elif 'splitkv' in kname:
        t = 'splitkv'
    else:
        t = 'fwd_prefill'
    
    by_type[t]['count'] += 1
    if col_l2_read_hit and col_l2_read_hit < len(row):
        by_type[t]['l2_read'].append(clean_num(row[col_l2_read_hit]))
    if col_l2_write_hit and col_l2_write_hit < len(row):
        by_type[t]['l2_write'].append(clean_num(row[col_l2_write_hit]))
    if col_dram_throughput and col_dram_throughput < len(row):
        by_type[t]['dram'].append(clean_num(row[col_dram_throughput]))
    if col_sm_throughput and col_sm_throughput < len(row):
        by_type[t]['sm'].append(clean_num(row[col_sm_throughput]))
    if col_occupancy and col_occupancy < len(row):
        by_type[t]['occ'].append(clean_num(row[col_occupancy]))
    if col_duration and col_duration < len(row):
        by_type[t]['dur'].append(clean_num(row[col_duration]))

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

for t, d in sorted(by_type.items()):
    print(f"\n  {t} ({d['count']} kernels):")
    print(f"    Avg duration:     {avg(d['dur']):.1f} us")
    print(f"    Avg L2 read hit:  {avg(d['l2_read']):.1f}%")
    print(f"    Avg L2 write hit: {avg(d['l2_write']):.1f}%")
    print(f"    Avg DRAM TP:      {avg(d['dram']):.1f}% of peak")
    print(f"    Avg SM TP:        {avg(d['sm']):.1f}% of peak")
    print(f"    Avg Occupancy:    {avg(d['occ']):.1f}%")
