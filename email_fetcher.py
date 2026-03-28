import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from config import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


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

        emails.append({
            'id': msg['id'],
            'snippet': msg_data.get('snippet', ''),
            'from': from_header,
        })

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
