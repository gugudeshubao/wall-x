#!/usr/bin/env python3
"""Convert Markdown article to WeChat-compatible HTML with inline styles."""
import re
import sys
import html

# Style constants
STYLES = {
    'body': "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 16px; color: #333; line-height: 1.8; padding: 0 10px;",
    'h1': "font-size: 24px; font-weight: bold; color: #1a1a1a; text-align: center; margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #333;",
    'h2': "font-size: 20px; font-weight: bold; color: #1a1a1a; margin: 28px 0 16px; padding-left: 10px; border-left: 4px solid #ff6600;",
    'h3': "font-size: 18px; font-weight: bold; color: #1a1a1a; margin: 24px 0 12px;",
    'h4': "font-size: 16px; font-weight: bold; color: #1a1a1a; margin: 20px 0 10px;",
    'p': "margin: 10px 0; text-align: justify;",
    'strong': "font-weight: bold; color: #1a1a1a;",
    'code': "background: #f0f0f0; color: #e83e8c; font-family: 'SF Mono', Menlo, monospace; font-size: 14px; padding: 2px 6px; border-radius: 3px;",
    'pre': "display: block; margin: 14px 0; padding: 14px; background: #1e1e1e; color: #d4d4d4; font-family: 'SF Mono', 'Fira Code', Menlo, monospace; font-size: 13px; line-height: 1.6; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;",
    'blockquote': "margin: 16px 0; padding: 12px 16px; background: #f7f7f7; border-left: 4px solid #ff6600; color: #666; font-size: 15px;",
    'hr': "border: none; border-top: 1px solid #ddd; margin: 24px 0;",
    'table': "width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px;",
    'th': "background: #f5f5f5; font-weight: bold; text-align: left; padding: 8px 10px; border: 1px solid #ddd;",
    'td': "padding: 8px 10px; border: 1px solid #ddd; text-align: left;",
    'ul': "margin: 10px 0; padding-left: 24px;",
    'ol': "margin: 10px 0; padding-left: 24px;",
    'li': "margin: 6px 0;",
    'section': "margin: 16px 0; padding: 16px; background: #fff8f0; border: 1px solid #ff6600; border-radius: 6px;",
    'footer': "margin: 30px 0 10px; padding: 16px; background: #f9f9f9; border-top: 2px solid #ddd; font-size: 13px; color: #999; line-height: 1.6;",
}

def escape(text):
    return html.escape(text, quote=False)

def process_inline(text):
    """Process inline markdown: **bold**, `code`, [link](url)"""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f'<strong style="{STYLES["strong"]}">{m.group(1)}</strong>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', lambda m: f'<code style="{STYLES["code"]}">{escape(m.group(1))}</code>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text

def convert_table(lines):
    """Convert markdown table lines to HTML table."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    if len(rows) < 2:
        return ''
    # Skip separator row (row[1] with ---)
    header = rows[0]
    data_rows = rows[2:] if len(rows) > 2 else []
    
    out = f'<table style="{STYLES["table"]}">\n'
    out += '<tr>' + ''.join(f'<th style="{STYLES["th"]}">{process_inline(c)}</th>' for c in header) + '</tr>\n'
    for row in data_rows:
        out += '<tr>' + ''.join(f'<td style="{STYLES["td"]}">{process_inline(c)}</td>' for c in row) + '</tr>\n'
    out += '</table>\n'
    return out

def convert_md_to_html(md_text):
    lines = md_text.split('\n')
    out = []
    out.append('<!DOCTYPE html>')
    out.append('<html>')
    out.append('<head><meta charset="utf-8"><title>WeChat Article</title></head>')
    out.append(f'<body style="{STYLES["body"]}">')
    
    i = 0
    in_tldr = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Empty line
        if not stripped:
            i += 1
            continue
        
        # Code block
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = escape('\n'.join(code_lines))
            out.append(f'<pre style="{STYLES["pre"]}">{code_text}</pre>')
            continue
        
        # Table
        if '|' in stripped and stripped.startswith('|'):
            table_lines = []
            while i < len(lines) and '|' in lines[i].strip() and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            out.append(convert_table(table_lines))
            continue
        
        # Headings
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title = process_inline(stripped[2:])
            out.append(f'<h1 style="{STYLES["h1"]}">{title}</h1>')
            i += 1
            continue
        if stripped.startswith('## '):
            title = process_inline(stripped[3:])
            out.append(f'<h2 style="{STYLES["h2"]}">{title}</h2>')
            i += 1
            continue
        if stripped.startswith('### '):
            title = process_inline(stripped[4:])
            out.append(f'<h3 style="{STYLES["h3"]}">{title}</h3>')
            i += 1
            continue
        if stripped.startswith('#### '):
            title = process_inline(stripped[5:])
            out.append(f'<h4 style="{STYLES["h4"]}">{title}</h4>')
            i += 1
            continue
        
        # HR
        if stripped == '---':
            out.append(f'<hr style="{STYLES["hr"]}"/>')
            i += 1
            continue
        
        # TL;DR section (special formatting)
        if stripped.startswith('**TL;DR**'):
            tldr_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('---') and not lines[i].strip().startswith('#'):
                tldr_lines.append(lines[i].strip())
                i += 1
            content = '<br/>'.join(process_inline(l) for l in tldr_lines)
            out.append(f'<section style="{STYLES["section"]}">{content}</section>')
            continue
        
        # Blockquote
        if stripped.startswith('>'):
            bq_lines = []
            while i < len(lines) and (lines[i].strip().startswith('>') or (lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('---') and not lines[i].strip().startswith('|') and bq_lines)):
                text = lines[i].strip()
                if text.startswith('> '):
                    text = text[2:]
                elif text.startswith('>'):
                    text = text[1:]
                # Check if next line continues the blockquote
                bq_lines.append(text)
                i += 1
                if i < len(lines) and not lines[i].strip().startswith('>') and not lines[i].strip().startswith('-'):
                    break
            content = '<br/>'.join(process_inline(l) for l in bq_lines if l)
            out.append(f'<blockquote style="{STYLES["blockquote"]}">{content}</blockquote>')
            continue
        
        # Unordered list
        if stripped.startswith('- '):
            list_items = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                item = process_inline(lines[i].strip()[2:])
                list_items.append(f'<li style="{STYLES["li"]}">{item}</li>')
                i += 1
            out.append(f'<ul style="{STYLES["ul"]}">{"".join(list_items)}</ul>')
            continue
        
        # Ordered list
        if re.match(r'^\d+\.', stripped):
            list_items = []
            while i < len(lines) and re.match(r'^\d+\.', lines[i].strip()):
                item = process_inline(re.sub(r'^\d+\.\s*', '', lines[i].strip()))
                list_items.append(f'<li style="{STYLES["li"]}">{item}</li>')
                i += 1
            out.append(f'<ol style="{STYLES["ol"]}">{"".join(list_items)}</ol>')
            continue
        
        # Footer (italic text at end)
        if stripped.startswith('*') and stripped.endswith('*') and len(stripped) > 100:
            content = process_inline(stripped[1:-1])
            out.append(f'<p style="{STYLES["footer"]}">{content}</p>')
            i += 1
            continue
        
        # Regular paragraph
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('---') and not lines[i].strip().startswith('```') and not lines[i].strip().startswith('|') and not lines[i].strip().startswith('>') and not lines[i].strip().startswith('- ') and not re.match(r'^\d+\.', lines[i].strip()) and not lines[i].strip().startswith('**TL;DR'):
            para_lines.append(lines[i].strip())
            i += 1
        if para_lines:
            content = process_inline(' '.join(para_lines))
            out.append(f'<p style="{STYLES["p"]}">{content}</p>')
            continue
        
        i += 1
    
    out.append('</body>')
    out.append('</html>')
    return '\n'.join(out)

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.md', '_wechat.html')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        md = f.read()
    
    html_content = convert_md_to_html(md)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated: {output_file} ({len(html_content)} chars)")
