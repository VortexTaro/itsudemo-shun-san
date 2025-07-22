import os.path
import pickle
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Google Sheetsの読み書き権限
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# 書き込み対象のスプレッドシートIDとシート名を指定
SPREADSHEET_ID = '1snuaHGKyLkET-2AnOuEH6W29Rr2DD8PHBTTAIZNV56w'
TARGET_SHEET_NAME = 'スピ・自己啓発' # 書き込み対象のシート名

# 認証情報と読み込むマークダウンファイル
TOKEN_FILE = 'token_sheets_write.pickle'
CREDENTIALS_FILE = 'credentials.json'
MARKDOWN_FILE = 'コンセプトメイキング/広告で勝ち確なコンセプト.md'

def parse_markdown(file_path):
    """マークダウンファイルを解析し、スプレッドシート用のデータ構造に変換する"""
    print(f"-> マークダウンファイルを解析中: {file_path}")
    data_for_sheet = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                data_for_sheet.append(['']) # 空行を追加
                continue
            
            # ヘッダー行の場合、`#` とスペースを削除して1列目に追加
            if line.startswith('#'):
                cleaned_header = re.sub(r'^#+\s*', '', line)
                data_for_sheet.append([cleaned_header])
                continue

            # リストアイテムを処理
            if line.startswith('- '):
                line_content = line[2:].strip()
                
                # ()の注釈を分離
                match = re.match(r'(.+?)\s*\((.+)\)', line_content)
                if match:
                    keyword = match.group(1).strip()
                    annotation = f"({match.group(2).strip()})"
                    data_for_sheet.append([keyword, annotation])
                else:
                    data_for_sheet.append([line_content])
                continue
            
            # それ以外の予期しない行は、念のためそのまま追加
            data_for_sheet.append([line])
                
    print(f"✓ 解析完了。総行数: {len(data_for_sheet)}")
    return data_for_sheet

def main():
    """マークダウンの内容をGoogleスプレッドシートに書き込む"""
    print('=== スプレッドシート書き込み処理開始 ===')
    
    # 認証処理
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
    print('✓ 認証完了（書き込み権限付き）')
    
    # マークダウンファイルを解析してデータ準備
    values = parse_markdown(MARKDOWN_FILE)

    try:
        # 1. シートをクリア
        print(f"-> '{TARGET_SHEET_NAME}' シートをクリア中...")
        service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID,
            range=TARGET_SHEET_NAME
        ).execute()
        print("✓ シートをクリアしました。")

        # 2. データを書き込み
        print(f"-> '{TARGET_SHEET_NAME}' シートにデータを書き込み中...")
        body = {
            'values': values
        }
        result = service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=TARGET_SHEET_NAME,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        
        print(f'✅ 書き込み完了!')
        print(f'   更新されたセル範囲: {result.get("updatedRange")}')

    except Exception as e:
        print(f'❌ エラーが発生しました: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 