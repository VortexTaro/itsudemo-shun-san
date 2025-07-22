import os
import sys
import argparse
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# --- 定数 ---
# KNOWLEDGE_BASE_DIR = "google_api/google_docs_source" # 検索対象を限定
FAISS_INDEX_PATH = "data/faiss_index"

def load_documents(directory, glob_pattern):
    """指定されたディレクトリから指定されたパターンのファイルを再帰的に読み込む"""
    print(f"現在の作業ディレクトリ: {os.getcwd()}")
    abs_directory = os.path.abspath(directory)
    print(f"'{abs_directory}' (絶対パス) から '{glob_pattern}' ファイルを読み込んでいます...")
    if not os.path.isdir(abs_directory):
        print(f"エラー: ディレクトリ '{abs_directory}' が見つかりません。")
        return []
    loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=TextLoader, use_multithreading=True, show_progress=True, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    print(f"このディレクトリから {len(documents)} 個のドキュメントが見つかりました。")
    return documents

def split_documents(documents):
    """ドキュメントを適切なサイズのチャンクに分割する"""
    print("ドキュメントをチャンクに分割しています...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"ドキュメントは {len(texts)} 個のチャンクに分割されました。")
    return texts

def create_and_save_faiss_index(texts, api_key):
    """OpenAIのEmbeddingsを使用してFAISSインデックスを作成し、保存する"""
    if not texts:
        print("テキストチャンクがありません。インデックスは作成されません。")
        return
        
    print("Embeddingモデルを読み込んでいます...")
    try:
        # Streamlitのsecretsからキーを渡すように修正
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        print("FAISSインデックスを作成しています...（ドキュメント数によっては時間がかかります）")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        print(f"インデックスの作成が完了し、'{FAISS_INDEX_PATH}' に保存されました。")
    except Exception as e:
        print(f"\n--- エラー ---")
        print(f"FAISSインデックスの作成中にエラーが発生しました: {e}")
        print("OpenAIのAPIキーが正しいか、また`.env`ファイルが正しく設定されているか確認してください。")
        sys.exit(1)


def main():
    """メインの処理フロー"""
    print("--- 知識ベース構築スクリプト開始 ---")
    
    # .envファイルの代わりにStreamlitのsecretsを模倣
    # このスクリプトは直接Streamlitから実行されないため、
    # 環境変数から読み込むか、別の方法でキーを渡す必要がある。
    # ここでは便宜上、環境変数から読み込む形を維持する。
    load_dotenv()
    
    # APIキーを環境変数から取得
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # .streamlit/secrets.tomlから読み込む試み
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            api_key = secrets.get("OPENAI_API_KEY")
            if not api_key:
                raise KeyError
        except (FileNotFoundError, KeyError):
             print("\n--- エラー ---")
             print("OpenAIのAPIキーが見つかりません。")
             print("`.streamlit/secrets.toml` に `OPENAI_API_KEY='あなたのAPIキー'` を設定するか、")
             print("`.env` ファイルを作成してそこに設定してください。")
             sys.exit(1)

    # 知識ベースのソースディレクトリとファイルパターンを定義
    KNOWLEDGE_SOURCES = [
        ("オーダーノート現実創造プログラム", "**/*.txt"),
        # ("docs", "**/*.md"), # docsは除外
        # ("Clippings", "**/*.md"), # Clippingsは除外
    ]
    
    # 全てのソースからドキュメントを読み込む
    all_documents = []
    print("\n--- ドキュメントの読み込み開始 ---")
    for directory, pattern in KNOWLEDGE_SOURCES:
        # 存在しないディレクトリはスキップ
        if not os.path.isdir(directory):
            print(f"警告: ディレクトリ '{directory}' が見つからないため、スキップします。")
            continue
        docs = load_documents(directory, pattern)
        all_documents.extend(docs)
    
    print(f"\n--- 全てのソースから合計 {len(all_documents)} 個のドキュメントを読み込みました ---")

    # 処理の実行
    texts = split_documents(all_documents)
    create_and_save_faiss_index(texts, api_key)
    
    print("\n--- 知識ベース構築完了！ ---")

if __name__ == "__main__":
    main() 