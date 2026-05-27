"""
alerts.py
---------
Alert Management — email notifications for anomalies/zero-days.

Configuration:
    Set EMAIL_ENABLED = True and configure SMTP credentials below to enable.
"""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# EMAIL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_ENABLED    = False  # Set to True after configuring SMTP credentials
SMTP_SERVER      = "smtp.gmail.com"
SMTP_PORT        = 587
SENDER_EMAIL     = ""        # Your sender email address
SENDER_PASSWORD  = ""        # App password or SMTP password
RECIPIENT_EMAIL  = ""        # Destination email address
ALERT_THRESHOLD  = 0.30  # Send alert if confidence > this

# ─────────────────────────────────────────────────────────────────────────────
# Alert Service
# ─────────────────────────────────────────────────────────────────────────────

def send_email_alert(src_ip: str, dst_ip: str, attack_type: str, confidence: float, 
                     protocol: str, is_zero_day: bool = False):
    """
    Send email alert for detected anomaly.
    
    Args:
        src_ip: Source IP
        dst_ip: Destination IP
        attack_type: Type of attack (ATTACK or ZERO-DAY)
        confidence: Detection confidence (0-1)
        protocol: Protocol (TCP, UDP, ICMP, etc.)
        is_zero_day: Whether it's a zero-day candidate
    """
    if not EMAIL_ENABLED:
        return False

    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("[WARN] Email alerts are enabled, but SMTP credentials are incomplete.")
        return False
    
    try:
        # Build email
        subject = f"🚨 NetGuard Alert: {attack_type} Detected" if not is_zero_day else f"⚠️ NetGuard ZERO-DAY Alert"
        
        body = f"""
╔════════════════════════════════════════════╗
║     NetGuard Security Alert              ║
╚════════════════════════════════════════════╝

🕐 Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Attack Type:  {attack_type}
⚡ Severity:     {'CRITICAL (Zero-Day)' if is_zero_day else 'HIGH'}
📊 Confidence:   {confidence:.2%}

🔍 Details:
  Source IP:    {src_ip}
  Target IP:    {dst_ip}
  Protocol:     {protocol.upper()}

⚠️  Action Required:
  - Review the dashboard: http://localhost:8501
  - Check detector logs for more context
  - Consider blocking source IP if malicious

---
NetGuard AI-Driven Anomaly Detection
"""
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"[ALERT] 📧 Email sent to {RECIPIENT_EMAIL}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


def should_alert(confidence: float, is_zero_day: bool) -> bool:
    """Check if alert should be sent based on confidence/zero-day."""
    if is_zero_day:
        return True  # Always alert on zero-days
    return confidence >= ALERT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Setup Instructions
# ─────────────────────────────────────────────────────────────────────────────

SETUP_INSTRUCTIONS = """
═══════════════════════════════════════════════════════════════════════════════
                        EMAIL ALERTS SETUP GUIDE
═══════════════════════════════════════════════════════════════════════════════

To enable email alerts, follow these steps:

1. GMAIL SETUP (if using Gmail):
   a) Go to: https://myaccount.google.com/apppasswords
   b) Select "Mail" and "Windows Computer"
   c) Copy the 16-character app password (NOT your regular Gmail password)

2. UPDATE alerts.py:
   - Set EMAIL_ENABLED = True
   - Set SENDER_EMAIL = "your_gmail@gmail.com"
   - Set SENDER_PASSWORD = "xxxx xxxx xxxx xxxx" (16-char app password)
   - Set RECIPIENT_EMAIL = "admin@example.com" (where to send alerts)
   - Adjust ALERT_THRESHOLD (0.85 = 85% confidence required)

3. OTHER EMAIL PROVIDERS:
   Gmail:     SMTP_SERVER = "smtp.gmail.com",     SMTP_PORT = 587
   Outlook:   SMTP_SERVER = "smtp-mail.outlook.com", SMTP_PORT = 587
   Yahoo:     SMTP_SERVER = "smtp.mail.yahoo.com",   SMTP_PORT = 587

4. TEST:
   python -c "from alerts import send_email_alert; send_email_alert('192.168.1.1', '192.168.1.6', 'TCP SYN Flood', 0.95, 'tcp')"

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
