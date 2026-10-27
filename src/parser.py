# src/parser.py
import re
from datetime import datetime

# Define regex patterns to match different log formats
LOG_PATTERNS = [
    # Example: 2025-09-24 14:12:06,145 ERROR database - Connection failed ...
    r'(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?)\s+(?P<level>[A-Z]+)\s+(?P<service>[\w\-.]+)\s*[:\-]\s*(?P<msg>.*)',
    # Example: 24/09/2025 14:12:06 ERROR database - Connection failed ...
    r'(?P<ts>\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+(?P<level>[A-Z]+)\s+(?P<service>[\w\-.]+)\s*[:\-]\s*(?P<msg>.*)',
]

def parse_line(line: str) -> dict:
    line = line.rstrip("\n")
    for pat in LOG_PATTERNS:
        m = re.match(pat, line)
        if m:
            d = m.groupdict()
            ts = d.get('ts')
            parsed_ts = None
            if ts:
                try:
                    parsed_ts = datetime.fromisoformat(ts.replace(',', '.'))
                except Exception:
                    try:
                        parsed_ts = datetime.strptime(ts, '%d/%m/%Y %H:%M:%S')
                    except Exception:
                        parsed_ts = None
            return {
                "timestamp": parsed_ts, "level": d.get('level'),
                "service": d.get('service'), "msg": d.get('msg', '').strip(),
                "raw": line
            }
    # Fallback: return raw line only
    return {
        "timestamp": None, "level": None, "service": None,
        "msg": line.strip(), "raw": line
    }

def parse_file(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                yield parse_line(line)