import streamlit as st
import os
import json
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from datetime import datetime
import streamlit.components.v1 as components
import asyncio
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
import glob

def generate_source_reasons(client, prompt, docs_with_scores):
    """
    各参照ドキュメントがユーザーのプロンプトに本当に関連しているかを判断し、
    関連している場合はその理由を、していない場合はその旨を返します。
    """
    if not docs_with_scores:
        return []

    content_list = []
    for i, (doc, score) in enumerate(docs_with_scores):
        content_list.append(f"<{i+1}>\\n{doc.page_content}\\n</{i+1}>")
    
    formatted_chunks = "\\n\\n".join(content_list)

    system_prompt = "あなたは、ユーザーの質問と複数のテキスト断片の関係性を分析する専門家です。各テキスト断片がユーザーの質問に本当に関連しているかを厳密に判断し、関連している場合はその核心的な理由を、関連していない場合はその旨を明確に示してください。"
    
    user_message = f"""以下のユーザーの質問と、それに関連する可能性のあるテキスト断片リストを読んでください。

# ユーザーの質問
{prompt}

# テキスト断片リスト
{formatted_chunks}

# 指示
各テキスト断片について、以下のどちらかの形式で回答してください。
1.  **本当に関連している場合:** `理由: [ここに具体的な理由を1文で記述]`
2.  **関連していない場合:** `理由: [IRRELEVANT]`

番号を付けてリスト形式で出力し、他の余計な言葉は含めないでください。

例:
1. 理由: 〇〇という課題に対する具体的な解決策が示されているため。
2. 理由: [IRRELEVANT]
3. 理由: 質問内のキーワード「△△」に関する詳細な背景を説明しているため。
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=system_prompt + "\\n\\n" + user_message,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=500,
            )
        )
        reasons_text = response.text
        reasons = []
        for line in reasons_text.strip().split('\\n'):
            if '理由: ' in line:
                reason = line.split('理由: ', 1)[1].strip()
                reasons.append(reason)
        
        if len(reasons) == len(docs_with_scores):
            return reasons
        else:
            # パース失敗時は全て無関係として扱う
            return ["[IRRELEVANT]"] * len(docs_with_scores)
    except Exception:
        return ["[IRRELEVANT]"] * len(docs_with_scores)


# --- 定数 ---
SIMILARITY_THRESHOLD = 0.7 # 類似度評価のしきい値を再度有効化
FEEDBACK_LOG_FILE = "feedback.log"

def log_feedback(message_id, user_prompt, assistant_response, feedback):
    """
    会話のフィードバックをファイルに記録します。
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "message_id": message_id,
        "user_prompt": user_prompt,
        "assistant_response": assistant_response,
        "feedback": feedback
    }
    with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\\n")

# --- 初期設定 ---
st.title("いつでもしゅんさん")

# StreamlitのsecretsからAPIキーを設定
try:
    api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError("API key not found")
    client = genai.Client(api_key=api_key)

    # 非同期処理の初期化問題を解決するため、グローバルスコープでイベントループを設定
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        
    # 埋め込みモデルを一元管理
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
except KeyError:
    st.error("Gemini APIキーが設定されていません。Streamlit Cloudの設定でGEMINI_API_KEYまたはGOOGLE_API_KEYを追加してください。")
    st.stop()

# --- 知識ベース構築 ---
KNOWLEDGE_SOURCES = [
    ("引き寄せアプリ/オーダーノート現実創造プログラム", "**/*.txt"),
]
FAISS_INDEX_PATH = "data/faiss_index_v2" # 新しいパスに変更

def build_and_save_faiss_index(embeddings):
    st.info("知識ベースを再構築しています...")
    try:
        source_directory, pattern = KNOWLEDGE_SOURCES[0]
        
        # globを直接使い、再帰的にファイルをリストアップする確実な方法に変更
        search_path = os.path.join(source_directory, pattern)
        all_file_paths = glob.glob(search_path, recursive=True)

        if not all_file_paths:
            st.error(f"'{source_directory}' 内にドキュメントが見つかりませんでした。globパターン: '{search_path}'")
            st.stop()

        # 個別のファイルをTextLoaderで読み込む
        documents = []
        for file_path in all_file_paths:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"ファイル '{file_path}' の読み込み中にエラー: {e}")
        
        st.info(f"'{source_directory}' から {len(documents)} 件のドキュメントを読み込み、チャンクに分割しました。")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        st.info("FAISSインデックスの構築と保存を開始します...")
        db = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        db.save_local(FAISS_INDEX_PATH)
        st.success(f"新しい知識ベースを '{FAISS_INDEX_PATH}' に保存しました。")
        return db
    except Exception as e:
        st.error(f"FAISSインデックスの構築中に致命的なエラーが発生しました: {e}\\n{traceback.format_exc()}")
        st.stop()


# --- FAISSインデックスのロード ---
@st.cache_resource
def load_faiss_index(path, _embeddings):
    """FAISSインデックスをロードする（存在しないか壊れていれば再構築）"""
    try:
        # まずイベントループを確実に設定
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if os.path.exists(path):
        try:
            return FAISS.load_local(path, _embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"既存の知識ベースの読み込みに失敗しました ({e})。再構築を試みます。")
            return build_and_save_faiss_index(_embeddings)
    else:
        return build_and_save_faiss_index(_embeddings)

db = load_faiss_index(FAISS_INDEX_PATH, embeddings)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """僕はしゅんさんのクローンです。しゅんさんが教えてくれた情報を元にあなたの質問に答えちゃうよ！引き寄せの法則・オーダーノートを学ぶ中で疑問や人生相談などあればなんなりチャットから教えてください！

※あなたが質問したことはいかなることであっても、しゅんさんや他の人には見えないから、安心してね！""",
        "id": str(uuid.uuid4()),
    }]
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = {}

# --- チャット履歴の表示 ---
for i, msg in enumerate(st.session_state.messages):
    avatar_url = "assets/avatar.png" if msg["role"] == "assistant" else "user"
    with st.chat_message(msg["role"], avatar=avatar_url):
        st.markdown(msg["content"])
        
        # --- フィードバック機能 ---
        # AIの応答で、かつ最初の挨拶ではない場合にボタンを表示
        if msg["role"] == "assistant" and i > 0:
            feedback_status = st.session_state.feedback_given.get(msg["id"])
            
            if not feedback_status:
                cols = st.columns([1, 1, 8])
                with cols[0]:
                    if st.button("👍", key=f"good_{msg['id']}", help="この回答に満足"):
                        user_prompt = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_prompt = st.session_state.messages[i-1]["content"]
                        log_feedback(msg['id'], user_prompt, msg['content'], "good")
                        st.session_state.feedback_given[msg['id']] = 'good'
                        st.toast("フィードバックをありがとうございます！")
                        st.rerun()
                
                with cols[1]:
                    if st.button("👎", key=f"bad_{msg['id']}", help="この回答に不満"):
                        user_prompt = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_prompt = st.session_state.messages[i-1]["content"]
                        log_feedback(msg['id'], user_prompt, msg['content'], "bad")
                        st.session_state.feedback_given[msg['id']] = 'bad'
                        st.toast("改善のためのフィードバック、感謝します。")
                        st.rerun()
            else:
                st.markdown(
                    f"<span style='color: #4A4A4A;'>{'👍' if feedback_status == 'good' else '👎'} フィードバック済み</span>", 
                    unsafe_allow_html=True
                )

        # アシスタントのメッセージで、かつ参照元情報がある場合
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("参照元ファイル"):
                for item in msg["sources"]:
                    doc = item["doc"]
                    score = item["score"]
                    reason = item["reason"]
                    
                    st.markdown(f"**ファイル名:** `{doc.metadata.get('source', 'N/A')}`")
                    st.markdown(f"**選択理由:** {reason} (類似度スコア: {score:.4f})")
                    
                    # コンテナを使って参照箇所をスクロール可能に
                    content_html = f"""
                        <div style="background-color: #262730; border-radius: 5px; padding: 10px; height: 150px; overflow-y: auto; border: 1px solid #333;">
                            <pre style="white-space: pre-wrap; word-wrap: break-word; color: #FAFAFA; font-family: 'Source Code Pro', monospace;">{doc.page_content}</pre>
                        </div>
                    """
                    components.html(content_html, height=170)
                    st.divider()

# --- ユーザーからの入力 ---
if prompt := st.chat_input("ここにメッセージを入力してください"):
    # ユーザーのメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    
    # 画面にユーザーのメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- AIの応答生成 ---
    with st.chat_message("assistant", avatar="assets/avatar.png"):
        message_placeholder = st.empty()
        full_response = ""

        # --- 知識ベースから関連情報を検索 ---
        context = ""
        source_docs_with_reasons = []
        is_relevant_info_found = False
        if db:
            try:
                # 検索を実行する現在のスレッドにイベントループを設定
                try:
                    asyncio.get_running_loop()
                except RuntimeError:  # 'RuntimeError: There is no current event loop...'
                    asyncio.set_event_loop(asyncio.new_event_loop())
                
                # ステップ1: 類似度スコアに基づき、候補を検索
                docs_with_scores = db.similarity_search_with_score(prompt, k=5)
                
                # 関連性チェックを簡略化し、類似度スコアとしきい値でフィルタリング
                relevant_sources = []
                if docs_with_scores:
                    for doc, score in docs_with_scores:
                        if score <= SIMILARITY_THRESHOLD: # しきい値よりスコアが低い（=類似度が高い）
                            relevant_sources.append({
                                "doc": doc,
                                "score": score,
                                "reason": "プロンプトとの類似性が見つかりました。" # 固定の理由
                            })
                
                # 関連するものが1つでもあれば、コンテキストを構築
                if relevant_sources:
                    is_relevant_info_found = True
                    source_docs_with_reasons = relevant_sources
                    context += "--- 関連情報 ---\n"
                    for item in source_docs_with_reasons:
                        context += item["doc"].page_content + "\n\n"

            except Exception as e:
                tb_str = traceback.format_exc()
                st.error(f"知識ベースの検索中にエラーが発生しました:\\n\\n```\\n{e}\\n\\n{tb_str}\\n```")

        # --- システムプロンプトの準備 ---
        try:
            with open("docs/system_prompt.md", "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            st.warning("docs/system_prompt.md が見つかりません。")
            system_prompt = "You are a helpful assistant."

        final_system_prompt = system_prompt
        # 関連情報が見つかった場合のみ、その情報をプロンプトに追加
        if is_relevant_info_found:
            final_system_prompt += "\n\n" + context
        else:
            # 見つからない場合は「なし」という明確な信号を送る
            final_system_prompt += "\n\n--- 関連情報 ---\nなし"

        # --- API呼び出しとストリーミング ---
        try:
            # メッセージ履歴をGoogleのAPIが期待する形式に変換
            history = []
            for m in st.session_state.messages[:-1]: # 最後のユーザーメッセージは除く
                role = 'user' if m['role'] == 'user' else 'model'
                history.append({'role': role, 'parts': [{'text': m['content']}]})

            # 新しいチャットセッションを開始
            chat = client.model(model_name="gemini-2.5-pro").start_chat(
                history=history,
            )
            
            # システムプロンプトを送信（Googleの新しい形式では直接サポートされないため、
            # ユーザーメッセージの先頭にコンテキストとして含める）
            prompt_with_context = f"{final_system_prompt}\n\nuser: {prompt}\nassistant:"

            # Geminiで応答を生成
            response = chat.send_message(
                prompt_with_context,
                generation_config=types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1500, # トークン上限を少し増やす
                )
            )
            
            # 安全性によるブロックを確認
            if not response.candidates:
                 full_response = "AIの応答が空でした。再度お試しください。"
            elif response.candidates[0].finish_reason == 'SAFETY':
                full_response = "申し訳ありません、安全上の理由により、この質問に対する応答を生成できませんでした。別の聞き方でお試しください。"
            else:
                full_response = response.text
                
            message_placeholder.markdown(full_response)

        except Exception as e:
            tb_str = traceback.format_exc()
            st.error(f"AIとの通信中にエラーが発生しました: {e}\n\n{tb_str}")
            full_response = "申し訳ありません、応答を生成できませんでした。"

    # --- 完全な応答をセッション履歴に追加 ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_docs_with_reasons,
        "id": str(uuid.uuid4()),
    })

    # --- 自動スクロールと再実行 ---
    st.session_state.scroll_to_bottom = True
    st.rerun()

# --- 自動スクロールの実行 ---
if st.session_state.get("scroll_to_bottom"):
    components.html(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        height=0
    )
    st.session_state.scroll_to_bottom = False 