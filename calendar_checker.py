import os
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from config import GMAIL_CREDENTIALS_FILE, GOOGLE_CALENDAR_ID

CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CALENDAR_TOKEN_FILE = 'calendar_token.json'

FOCUS_KEYWORDS = ['focus', 'deep work', 'coding', 'lecture', 'study']


def is_focus_block_now(calendar_service) -> bool:
    now = datetime.utcnow()
    time_min = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    time_max = (now + timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

    events_result = calendar_service.events().list(
        calendarId=GOOGLE_CALENDAR_ID,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    for event in events:
        title = event.get('summary', '').lower()
        if any(keyword in title for keyword in FOCUS_KEYWORDS):
            return True

    return False


def build_calendar_service():
    creds = None

    if os.path.exists(CALENDAR_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(CALENDAR_TOKEN_FILE, CALENDAR_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_CREDENTIALS_FILE, CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(CALENDAR_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)
    return service
