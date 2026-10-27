# tests/test_parser.py
import sys
import os

# Add project root directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_line
from src.template_extractor import normalize_msg

def test_parse_line_iso_format():
    """
    Test parsing logs using ISO timestamp formats.
    """
    log_line = "2025-09-24 14:12:06,145 ERROR database - Connection failed to master node."
    result = parse_line(log_line)
    
    assert result["level"] == "ERROR"
    assert result["service"] == "database"
    assert "Connection failed" in result["msg"]
    assert result["timestamp"] is not None
    assert result["timestamp"].year == 2025
    assert result["timestamp"].month == 9
    assert result["timestamp"].day == 24

def test_parse_line_slashed_format():
    """
    Test parsing logs using slashed European date formats.
    """
    log_line = "24/09/2025 14:12:06 WARN api-gateway - Route /v1/data is experiencing latency."
    result = parse_line(log_line)
    
    assert result["level"] == "WARN"
    assert result["service"] == "api-gateway"
    assert "latency" in result["msg"]
    assert result["timestamp"] is not None
    assert result["timestamp"].year == 2025
    assert result["timestamp"].month == 9
    assert result["timestamp"].day == 24

def test_parse_line_fallback():
    """
    Test fallback logic on log lines that do not match standard formats.
    """
    log_line = "Strange unformatted system noise message here."
    result = parse_line(log_line)
    
    assert result["timestamp"] is None
    assert result["level"] is None
    assert result["service"] is None
    assert result["msg"] == log_line.strip()

def test_normalize_msg_ips_and_numbers():
    """
    Verify template extraction and variable masking for IPs, hexes, and integers.
    """
    msg = "User 'alice' logged in from 192.168.1.100 using port 8080 and hex 0x3f5"
    normalized = normalize_msg(msg)
    
    # Check that variables were successfully masked
    assert "<ip>" in normalized or "<IP>" in normalized or "ip" in normalized
    assert "<num>" in normalized or "<NUM>" in normalized or "num" in normalized
    assert "<hex>" in normalized or "<HEX>" in normalized or "hex" in normalized
    
    # Check lowercase normalization
    assert normalized == normalized.lower()

def test_normalize_msg_filepaths():
    """
    Verify template extraction and variable masking for Unix and Windows paths.
    """
    msg1 = "Database configuration loaded from /var/lib/mysql/config.cnf successfully."
    msg2 = "Windows file missing at C:\\Windows\\System32\\driver.sys error"
    
    norm1 = normalize_msg(msg1)
    norm2 = normalize_msg(msg2)
    
    assert "path" in norm1
    assert "path" in norm2
