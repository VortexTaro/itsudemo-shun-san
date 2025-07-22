import os.path
import pickle
import re
import glob
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Google Sheetsの読み書き権限を要求
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets'  # 読み書き権限
]

SPREADSHEET_ID = '1EzngFbCpoa4KVfls_9VRQOrKahoLYch3jzLIW58-5aY'
TOKEN_FILE = 'token_sheets_write.pickle'  # 新しいトークンファイル
CREDENTIALS_FILE = 'credentials.json'

def main():
    """処理済み行のI列にチェックマークを追加"""
    print('=== スプレッドシート書き込み認証開始 ===')
    
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
    
    # 作成されたファイルから処理済み行番号を抽出
    processed_rows = set()
    for filename in glob.glob('row*.txt'):
        match = re.match(r'row(\d+)_', filename)
        if match:
            row_num = int(match.group(1))
            processed_rows.add(row_num)

    processed_rows = sorted(list(processed_rows))
    print(f'処理済み行番号: {processed_rows[:10]}...（総数: {len(processed_rows)}）')

    # I列にチェックマークを入れる準備
    updates = []
    for row_num in processed_rows:
        updates.append({
            'range': f'I{row_num}',
            'values': [['✓']]
        })

    print(f'更新予定のセル数: {len(updates)}')

    try:
        # バッチでスプレッドシートを更新
        body = {
            'valueInputOption': 'RAW',
            'data': updates
        }
        
        result = service.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID, 
            body=body
        ).execute()
        
        print(f'✅ 更新完了!')
        print(f'   更新されたセル数: {result.get("totalUpdatedCells", 0)}')
        print(f'   更新された行数: {result.get("totalUpdatedRows", 0)}')
        
        # 更新された行の詳細を表示
        print(f'\n更新された行（最初の10個）:')
        for i, row_num in enumerate(processed_rows[:10]):
            print(f'  行{row_num}: ✓')
        
        if len(processed_rows) > 10:
            print(f'  ... 他{len(processed_rows)-10}行も更新済み')
            
    except Exception as e:
        print(f'❌ 更新エラー: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 