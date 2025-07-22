import os.path
import pickle
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Google Sheets と Google Docs の両方の読み取り権限を要求
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/documents.readonly'
]

# 対象のスプレッドシートID
SPREADSHEET_ID = '1EzngFbCpoa4KVfls_9VRQOrKahoLYch3jzLIW58-5aY'
TOKEN_FILE = 'token_sheets.pickle'
CREDENTIALS_FILE = 'credentials.json'

def extract_document_id(url):
    """URLからGoogleドキュメントIDを抽出"""
    match = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else None

def extract_tab_id(url):
    """URLからタブIDを抽出"""
    match = re.search(r'tab=([a-zA-Z0-9._-]+)', url)
    return match.group(1) if match else None

def extract_text_from_content(content):
    """ドキュメントコンテンツからテキストを抽出"""
    text_parts = []
    
    for element in content:
        if 'paragraph' in element:
            para = element['paragraph']
            if 'elements' in para:
                for elem in para['elements']:
                    if 'textRun' in elem:
                        text_parts.append(elem['textRun'].get('content', ''))
        elif 'table' in element:
            table = element['table']
            for row in table.get('tableRows', []):
                for cell in row.get('tableCells', []):
                    for cell_content in cell.get('content', []):
                        if 'paragraph' in cell_content:
                            cell_para = cell_content['paragraph']
                            for elem in cell_para.get('elements', []):
                                if 'textRun' in elem:
                                    text_parts.append(elem['textRun'].get('content', ''))
    
    return ''.join(text_parts)

def safe_filename(text, max_length=50):
    """ファイル名として安全な文字列に変換"""
    text = re.sub(r'[^\w\s-]', '_', text)
    text = re.sub(r'\s+', '_', text)
    return text[:max_length]

def main():
    """メイン処理"""
    print('=== Google Sheets認証開始 ===')
    
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

    # Google SheetsとDocs APIサービスを初期化
    sheets_service = build('sheets', 'v4', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)
    
    print('✓ 認証完了')
    
    try:
        print('=== スプレッドシートからG列URLを取得中 ===')
        
        # G列からデータを取得
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='G:G'
        ).execute()
        
        values = result.get('values', [])
        print(f'G列の総行数: {len(values)}')
        
        # GoogleドキュメントURLを抽出
        doc_urls = []
        for i, row in enumerate(values):
            if row and len(row) > 0:
                cell_value = row[0]
                if 'docs.google.com/document' in str(cell_value):
                    doc_urls.append((i+1, cell_value))
        
        print(f'GoogleドキュメントURL数: {len(doc_urls)}')
        
        # 各URLの詳細を表示（最初の5個）
        for row_num, url in doc_urls[:5]:
            print(f'  行{row_num}: {str(url)[:80]}...')
        
        if len(doc_urls) > 5:
            print(f'  ... 他{len(doc_urls)-5}個のURL')
        
        print(f'\n=== {len(doc_urls)}個のドキュメントを処理開始 ===')
        
        processed_docs = {}  # 重複処理を避けるため
        total_tabs_saved = 0
        
        for row_num, url in doc_urls:
            print(f'\n--- 行{row_num}: 処理中 ---')
            
            doc_id = extract_document_id(url)
            if not doc_id:
                print('  ⚠️ ドキュメントIDが抽出できませんでした')
                continue
            
            # 既に処理済みかチェック
            if doc_id in processed_docs:
                print(f'  ✓ ドキュメント {doc_id} は既に処理済みです')
                continue
            
            processed_docs[doc_id] = True
            
            try:
                # ドキュメント全体を取得（全タブ含む）
                document = docs_service.documents().get(
                    documentId=doc_id,
                    includeTabsContent=True
                ).execute()
                
                doc_title = document.get('title', 'Untitled')
                print(f'  ドキュメント: {doc_title}')
                
                # タブ情報を取得
                tabs = document.get('tabs', [])
                print(f'  タブ数: {len(tabs)}')
                
                if not tabs:
                    print('  ⚠️ タブが見つかりませんでした')
                    continue
                
                # 各タブを処理
                for tab_index, tab in enumerate(tabs):
                    tab_properties = tab.get('tabProperties', {})
                    tab_id = tab_properties.get('tabId', f'tab_{tab_index}')
                    tab_title = tab_properties.get('title', f'タブ{tab_index+1}')
                    
                    print(f'    タブ {tab_index+1}: {tab_title}')
                    
                    # タブの内容を取得
                    doc_tab = tab.get('documentTab', {})
                    body = doc_tab.get('body', {})
                    content = body.get('content', [])
                    
                    # テキストを抽出
                    full_text = extract_text_from_content(content)
                    
                    # ファイル名を作成
                    safe_doc_title = safe_filename(doc_title)
                    safe_tab_title = safe_filename(tab_title)
                    filename = f'row{row_num:03d}_{safe_doc_title}_tab{tab_index+1}_{safe_tab_title}.txt'
                    
                    # ファイルに保存
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f'元スプレッドシート行: {row_num}\n')
                        f.write(f'ドキュメントタイトル: {doc_title}\n')
                        f.write(f'ドキュメントID: {doc_id}\n')
                        f.write(f'タブタイトル: {tab_title}\n')
                        f.write(f'タブID: {tab_id}\n')
                        f.write(f'タブインデックス: {tab_index}\n')
                        f.write(f'直接URL: https://docs.google.com/document/d/{doc_id}/edit?tab={tab_id}\n')
                        f.write(f'元URL: {url}\n')
                        f.write('='*80 + '\n\n')
                        f.write(full_text)
                    
                    total_tabs_saved += 1
                    print(f'      → {filename} (文字数: {len(full_text)})')
                    
            except Exception as e:
                print(f'  ❌ ドキュメント処理エラー: {e}')
                continue
        
        print(f'\n✅ 処理完了!')
        print(f'   処理済みドキュメント数: {len(processed_docs)}')
        print(f'   保存されたタブ数: {total_tabs_saved}')
        
    except Exception as e:
        print(f'❌ エラー: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 