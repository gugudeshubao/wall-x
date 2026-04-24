#!/usr/bin/env python3
"""Analyze nsys kernel summary CSV and classify GPU time by kernel category."""
import sys
import csv

def analyze(csv_path, label):
    with open(csv_path) as f:
        # Skip non-CSV lines until header
        lines = f.readlines()
    
    # Find CSV header line
    header_idx = None
    for i, line in enumerate(lines):
        if '"Time (%)"' in line or 'Time (%)' in line:
            header_idx = i
            break
    if header_idx is None:
        print(f"ERROR: Could not find CSV header in {csv_path}")
        return
    
    csv_lines = lines[header_idx:]
    reader = csv.DictReader(csv_lines)
    rows = list(reader)
    
    # Normalize column names (strip whitespace)
    clean_rows = []
    for r in rows:
        clean = {k.strip(): v.strip() for k, v in r.items()}
        clean_rows.append(clean)
    rows = clean_rows
    
    # Find the right column names
    time_col = None
    name_col = None
    inst_col = None
    for k in rows[0].keys():
        if 'Total Time' in k:
            time_col = k
        if k == 'Name':
            name_col = k
        if 'Instances' in k or 'Inst' in k:
            inst_col = k
    
    if not time_col or not name_col:
        print(f"Columns: {list(rows[0].keys())}")
        return
    
    def get_time(r):
        v = r[time_col].replace(',', '').replace('"', '')
        return int(v)
    
    def get_name(r):
        return r[name_col].replace('"', '')
    
    total = sum(get_time(r) for r in rows)
    total_inst = sum(int(r[inst_col].replace(',', '').replace('"', '')) for r in rows) if inst_col else 0
    
    print(f'=== {label} GPU Kernel Summary ===')
    print(f'Total GPU kernel time: {total/1e6:.1f} ms')
    print(f'Total unique kernel types: {len(rows)}')
    if inst_col:
        print(f'Total kernel instances: {total_inst}')
    print()
    
    # Classify kernels
    categories = {}
    for r in rows:
        name = get_name(r)
        t = get_time(r)
        nl = name.lower()
        
        if 'gemm' in nl or 'gemv' in nl or 'cutlass' in nl:
            cat = 'GEMM/GEMV'
        elif 'fmha' in nl or 'flash' in nl:
            cat = 'Attention (fmha/flash)'
        elif 'softmax' in nl:
            cat = 'Softmax'
        elif 'moe_permute' in nl or 'moe_recover' in nl or 'permute_topk' in nl:
            cat = 'MoE permute/unpermute'
        elif 'rope' in nl or 'rot_pos' in nl:
            cat = 'RoPE'
        elif 'window' in nl:
            cat = 'Window index'
        elif 'reduce' in nl:
            cat = 'Reduce'
        elif 'sort' in nl:
            cat = 'RadixSort'
        elif 'cat' in nl and 'copy' in nl:
            cat = 'CatArrayCopy'
        elif 'copy' in nl:
            cat = 'Copy/dtype conversion'
        elif 'elementwise' in nl:
            cat = 'Elementwise ops'
        elif 'fill' in nl:
            cat = 'Fill'
        elif 'index' in nl or 'gather' in nl or 'scatter' in nl:
            cat = 'Index/Gather/Scatter'
        elif 'nchw' in nl or 'nhwc' in nl:
            cat = 'Layout transform (NCHW/NHWC)'
        else:
            cat = 'Other'
        
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += t
    
    # Sort by time descending
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
    
    print(f'{"Category":<35} {"Time (ms)":>10} {"Pct":>7}')
    print('-' * 55)
    for cat, t in sorted_cats:
        print(f'{cat:<35} {t/1e6:>10.1f} {t/total*100:>6.1f}%')
    print('-' * 55)
    print(f'{"TOTAL":<35} {total/1e6:>10.1f} {100.0:>6.1f}%')
    
    # Top 10 individual kernels
    print(f'\nTop 10 Individual Kernels:')
    print(f'{"Pct":>6} {"Time(ms)":>10} {"Inst":>7}  {"Kernel"}')
    print('-' * 80)
    top = sorted(rows, key=lambda r: -get_time(r))[:10]
    for r in top:
        t = get_time(r)
        inst = r[inst_col].replace(',', '').replace('"', '') if inst_col else '?'
        name = get_name(r)
        # Truncate name
        if len(name) > 80:
            name = name[:77] + '...'
        print(f'{t/total*100:>5.1f}% {t/1e6:>10.1f} {inst:>7}  {name}')


if __name__ == '__main__':
    csv_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else csv_path
    analyze(csv_path, label)
