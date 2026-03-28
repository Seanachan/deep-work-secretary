import os
import re
import time
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from config import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def enrich_with_history(service, email: dict) -> dict:
    from_field = email.get('from', '')
    match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', from_field)
    if not match:
        email['message_frequency'] = 0
        email['time_since_last_reply'] = 168.0
        return email

    sender_email = match.group(0)

    try:
        freq_result = service.users().messages().list(
            userId='me',
            q=f'from:{sender_email} newer_than:7d'
        ).execute()
        email['message_frequency'] = len(freq_result.get('messages', []))
    except Exception:
        email['message_frequency'] = 0

    try:
        thread_result = service.users().messages().list(
            userId='me',
            q=f'from:{sender_email} OR to:{sender_email}',
            maxResults=1
        ).execute()
        messages = thread_result.get('messages', [])
        if messages:
            msg_data = service.users().messages().get(
                userId='me',
                id=messages[0]['id'],
                format='metadata'
            ).execute()
            internal_date_ms = int(msg_data.get('internalDate', 0))
            now_ms = int(time.time() * 1000)
            hours_since = (now_ms - internal_date_ms) / (1000 * 3600)
            email['time_since_last_reply'] = max(hours_since, 0.0)
        else:
            email['time_since_last_reply'] = 168.0
    except Exception:
        email['time_since_last_reply'] = 168.0

    return email


def fetch_email_snippets(service):
    result = service.users().messages().list(
        userId='me',
        labelIds=['UNREAD'],
        maxResults=5
    ).execute()

    messages = result.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='metadata',
            metadataHeaders=['From']
        ).execute()

        headers = msg_data.get('payload', {}).get('headers', [])
        from_header = next(
            (h['value'] for h in headers if h['name'] == 'From'),
            ''
        )

        email = {
            'id': msg['id'],
            'snippet': msg_data.get('snippet', ''),
            'from': from_header,
        }
        email = enrich_with_history(service, email)
        emails.append(email)

    return emails


def build_gmail_service():
    creds = None

    if os.path.exists(GMAIL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(GMAIL_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service
