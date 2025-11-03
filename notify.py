import os
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pyairtable import Table

load_dotenv()
# Airtable settings
AIRTABLE_BASE = os.getenv("AIRTABLE_BASE")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")
AIRTABLE_KEY = os.getenv("AIRTABLE_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL", "ops@bookleaf.example")


def update_airtable(isbn, status, confidence, issues, overlay_url, validation_message):
    table = Table(AIRTABLE_KEY.strip(), AIRTABLE_BASE.strip(), AIRTABLE_TABLE.strip())

    # Check if record already exists
    records = table.all(formula=f"{{Book ID}} = '{isbn}'")

    # Define valid issue types as allowed by Airtable
    valid_issue_types = {
        "Badge Overlap",
        "Safe Margin",
        "Low Resolution",
        "OCR Low Confidence",
        "Filename Error"
    }
    # Normalize issues to match allowed options
    filtered_issues = []
    for issue in issues or []:
        normalized = issue.strip()
        if normalized in valid_issue_types:
            filtered_issues.append(normalized)
    fields = {
        "Book ID": isbn,
        "Status": status,
        "Confidence Score": round(confidence, 2),
        "Issue Type": filtered_issues if filtered_issues else [],
        "Visual Annotations URL": overlay_url,
        "Correction Instructions": validation_message,
    }

    if records:
        record_id = records[0]["id"]
        table.update(record_id, fields)
        return {"id": record_id, "fields": fields}
    else:
        record = table.create(fields)
        return record


def send_email(isbn, status, issues, overlay_url, confidence, to_email):
    subject = f"Cover Validation Result — {isbn} ({status})"
    if status == "PASS":
        body = f"""
        <p>Hello,</p>
        <p>Status: ✅ PASS</p>
        <p>Your cover passed all checks.</p>
        <p>Confidence: {confidence:.1f}%</p>
        <p><a href="{overlay_url}">View Overlay</a></p>
        <p>Thank you,<br>BookLeaf Publishing</p>
        """
    else:
        body = f"""
        <p>Hello,</p>
        <p>Status: ❌ REVIEW NEEDED</p>
        <p>Issues: {issues}</p>
        <p>Confidence: {confidence:.1f}%</p>
        <p><a href="{overlay_url}">View Overlay</a></p>
        <p>Please correct and re-upload your file.</p>
        """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        print(f"Email send failed: {e}")