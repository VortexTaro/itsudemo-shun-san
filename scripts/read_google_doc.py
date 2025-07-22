
import os.path
import pickle
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']

# The ID of a sample document.
DOCUMENT_ID = '1OHnHmFoN8ec74ErafG5EepmSTWM5FTU6FSNcPOjMLcs'

def main():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('docs', 'v1', credentials=creds)

    # Retrieve the documents contents from the Docs service.
    document = service.documents().get(documentId=DOCUMENT_ID).execute()

    with open('document.json', 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=4)

    print('The title of the document is: {}'.format(document.get('title')))
    
    # Extract and save the text content
    content = document.get('body').get('content')
    doc_content = ""
    print(f"Total structural elements: {len(content)}")
    for i, value in enumerate(content):
        print(f"Processing element {i+1}/{len(content)}")
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                text_run = elem.get('textRun')
                if text_run:
                    text_content = text_run.get('content')
                    # print(f"  - Found text: {text_content[:30]}...")
                    doc_content += text_content
        elif 'table' in value:
            print("  - Found a table, processing...")
            table = value.get('table')
            for row in table.get('tableRows'):
                for cell in row.get('tableCells'):
                    for cell_content in cell.get('content'):
                        if 'paragraph' in cell_content:
                            elements = cell_content.get('paragraph').get('elements')
                            for elem in elements:
                                text_run = elem.get('textRun')
                                if text_run:
                                    text_content = text_run.get('content')
                                    # print(f"  - Found text in table: {text_content[:30]}...")
                                    doc_content += text_content
        else:
            print(f"  - Skipping non-paragraph/non-table element: {list(value.keys())}")


    with open('google_doc_content.txt', 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print('Document content has been saved to google_doc_content.txt')


if __name__ == '__main__':
    main() 