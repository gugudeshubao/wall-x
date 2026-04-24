#!/usr/bin/env python3
"""Convert markdown to WeChat Official Account compatible HTML with inline styles."""
import re
import sys
import html

# WeChat-compatible inline styles
STYLES = {
    'body': 'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 16px; color: #333; line-height: 1.8; padding: 0 10px;',
    'h1': 'font-size: 24px; font-weight: bold; color: #1a1a1a; text-align: center; margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #333;',
    'h2': 'font-size: 20px; font-weight: bold; color: #1a1a1a; margin: 28px 0 16px; padding-left: 10px; border-left: 4px solid #ff6600;',
    'h3': 'font-size: 17px; font-weight: bold; color: #333; margin: 22px 0 12px;',
    'p': 'margin: 10px 0; text-align: justify;',
    'blockquote': 'margin: 16px 0; padding: 12px 16px; background: #f7f7f7; border-left: 4px solid #ff6600; color: #666; font-size: 15px;',
    'code_block': 'display: block; margin: 14px 0; padding: 14px; background: #1e1e1e; color: #d4d4d4; font-family: "SF Mono", "Fira Code", Menlo, monospace; font-size: 13px; line-height: 1.6; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;',
    'code_inline': 'background: #f0f0f0; color: #e83e8c; font-family: "SF Mono", Menlo, monospace; font-size: 14px; padding: 2px 6px; border-radius: 3px;',
    'table': 'width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px;',
    'th': 'background: #f5f5f5; font-weight: bold; text-align: left; padding: 8px 10px; border: 1px solid #ddd;',
    'td': 'padding: 8px 10px; border: 1px solid #ddd; text-align: left;',
    'ul': 'margin: 10px 0; padding-left: 24px;',
    'ol': 'margin: 10px 0; padding-left: 24px;',
    'li': 'margin: 6px 0;',
    'hr': 'border: none; border-top: 1px solid #ddd; margin: 24px 0;',
    'strong': 'font-weight: bold; color: #1a1a1a;',
    'tldr_box': 'margin: 16px 0; padding: 16px; background: #fff8f0; border: 1px solid #ff6600; border-radius: 6px;',
    'img_caption': 'text-align: center; font-size: 13px; color: #999; margin-top: 6px;',
}


def process_inline(text):
    """Process inline markdown: bold, inline code, links."""
    # Inline code (before bold, to avoid conflicts)
    text = re.sub(r'`([^`]+)`', lambda m: f'<code style="{STYLES["code_inline"]}">{html.escape(m.group(1))}</code>', text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', lambda m: f'<strong style="{STYLES["strong"]}">{m.group(1)}</strong>', text)
    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" style="color: #ff6600; text-decoration: none;">\1</a>', text)
    return text


def convert_table(lines):
    """Convert markdown table lines to HTML table."""
    out = f'<table style="{STYLES["table"]}">\n'
    for i, line in enumerate(lines):
        line = line.strip().strip('|')
        cells = [c.strip() for c in line.split('|')]
        # Skip separator row (---|---|---)
        if all(re.match(r'^[-:]+$', c) for c in cells):
            continue
        tag = 'th' if i == 0 else 'td'
        style = STYLES[tag]
        out += '<tr>'
        for cell in cells:
            cell = process_inline(cell)
            out += f'<{tag} style="{style}">{cell}</{tag}>'
        out += '</tr>\n'
    out += '</table>\n'
    return out


def convert_md_to_html(md_text):
    """Main conversion function."""
    lines = md_text.split('\n')
    html_parts = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Code block (``` ... ```)
        if line.strip().startswith('```'):
            lang = line.strip()[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(html.escape(lines[i]))
                i += 1
            i += 1  # skip closing ```
            code_content = '\n'.join(code_lines)
            html_parts.append(f'<pre style="{STYLES["code_block"]}">{code_content}</pre>')
            continue

        # Table (| ... | ... |)
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            html_parts.append(convert_table(table_lines))
            continue

        # Horizontal rule
        if re.match(r'^---+\s*$', line.strip()):
            html_parts.append(f'<hr style="{STYLES["hr"]}"/>')
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = process_inline(heading_match.group(2))
            tag = f'h{level}'
            html_parts.append(f'<{tag} style="{STYLES[tag]}">{text}</{tag}>')
            i += 1
            continue

        # Blockquote
        if line.strip().startswith('>'):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote_lines.append(lines[i].strip().lstrip('>').strip())
                i += 1
            quote_text = '<br/>'.join(process_inline(l) for l in quote_lines if l)
            html_parts.append(f'<blockquote style="{STYLES["blockquote"]}">{quote_text}</blockquote>')
            continue

        # TL;DR detection (special box)
        if line.strip().startswith('**TL;DR**') or line.strip().startswith('**TL;DR'):
            tldr_lines = [process_inline(line.strip())]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                tldr_lines.append(process_inline(lines[i].strip()))
                i += 1
            tldr_content = '<br/>'.join(tldr_lines)
            html_parts.append(f'<section style="{STYLES["tldr_box"]}">{tldr_content}</section>')
            continue

        # Ordered list
        ol_match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
        if ol_match:
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i].strip()):
                m = re.match(r'^\d+\.\s+(.+)$', lines[i].strip())
                if m:
                    items.append(process_inline(m.group(1)))
                i += 1
            list_html = f'<ol style="{STYLES["ol"]}">'
            for item in items:
                list_html += f'<li style="{STYLES["li"]}">{item}</li>'
            list_html += '</ol>'
            html_parts.append(list_html)
            continue

        # Unordered list
        if re.match(r'^[-*]\s+', line.strip()):
            items = []
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
                m = re.match(r'^[-*]\s+(.+)$', lines[i].strip())
                if m:
                    items.append(process_inline(m.group(1)))
                i += 1
            list_html = f'<ul style="{STYLES["ul"]}">'
            for item in items:
                list_html += f'<li style="{STYLES["li"]}">{item}</li>'
            list_html += '</ul>'
            html_parts.append(list_html)
            continue

        # Regular paragraph
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') \
                and not lines[i].strip().startswith('|') and not lines[i].strip().startswith('```') \
                and not lines[i].strip().startswith('>') and not re.match(r'^---+\s*$', lines[i].strip()) \
                and not re.match(r'^[-*]\s+', lines[i].strip()) \
                and not re.match(r'^\d+\.\s+', lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        if para_lines:
            text = process_inline(' '.join(para_lines))
            html_parts.append(f'<p style="{STYLES["p"]}">{text}</p>')
            continue

        i += 1

    body = '\n'.join(html_parts)
    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>WeChat Article</title></head>
<body style="{STYLES['body']}">
{body}
</body>
</html>'''


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/Users/sam/project/github/wall-x/docs/zhihu_orin_5090_deployment.md'
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.md', '_wechat.html')

    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_content = convert_md_to_html(md_content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Converted: {input_file}")
    print(f"Output:    {output_file}")
    print(f"Size:      {len(html_content):,} chars")
    print(f"\nUsage: Open the HTML file in browser, Cmd+A to select all, then paste into WeChat editor.")
