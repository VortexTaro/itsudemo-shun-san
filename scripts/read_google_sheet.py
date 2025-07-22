import os.path
import pickle
import csv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import re

# スプレッドシートの読み取り権限を要求します。
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# 読み込むスプレッドシートのIDを指定します。
SPREADSHEET_ID = '17Dojv7PURq79CG6v5o3WtzRiGtkqDsk6_ZDHfzd39UA'
TOKEN_FILE = 'token_sheets.pickle'
CREDENTIALS_FILE = 'credentials.json'
OUTPUT_DIR = 'sheets_output' # CSVを保存するフォルダ

def sanitize_filename(filename):
    """ファイル名として無効な文字を削除または置換します。"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def main():
    """Googleスプレッドシートの全シートからデータを読み込み、個別のCSVとして保存します。"""
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    try:
        sheet_metadata = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets = sheet_metadata.get('sheets', '')

        # 出力用ディレクトリを作成
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        print(f"'{OUTPUT_DIR}' フォルダにCSVファイルを保存します。")

        for sheet in sheets:
            sheet_title = sheet.get("properties", {}).get("title", "Untitled")
            print(f"-> '{sheet_title}' シートを読み込み中...")

            # 各シートのデータを取得
            range_name = f"'{sheet_title}'" # シート全体を指定
            result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
            values = result.get('values', [])

            if not values:
                print(f"   '{sheet_title}' シートは空です。スキップします。")
                continue

            sanitized_title = sanitize_filename(sheet_title)
            output_filename = os.path.join(OUTPUT_DIR, f"{sanitized_title}.csv")
            
            with open(output_filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(values)
            print(f"   ✓ '{sheet_title}' の内容を {output_filename} に保存しました。")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == '__main__':
    main() 