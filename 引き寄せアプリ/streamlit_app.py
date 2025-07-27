import streamlit as st
import os
import json
import google.generativeai as genai
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
import re

def generate_search_query(prompt, conversation_history):
    """ユーザーのプロンプトと会話履歴から、FAISS検索に最適なクエリを生成する"""
    
    # 会話履歴を整形
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    # プロンプトテンプレート
    prompt_template = f"""
あなたは優秀な検索アシスタントです。以下の会話履歴と最後のユーザープロンプトを分析し、ベクトルデータベースから最も関連性の高い情報を引き出すための、簡潔かつ効果的な検索クエリを生成してください。

【制約条件】
- 最も重要なキーワードを抽出してください。
- 検索クエリ以外の余計な文章は含めないでください。
- ユーザーの意図を正確に反映してください。

【会話履歴】
{history_str}

【最後のユーザープロンプト】
{prompt}

【生成すべき検索クエリ】
"""
    
    try:
        # Geminiに検索クエリの生成を依頼
        response = model.generate_content(prompt_template)
        search_query = response.text.strip()
        # st.info(f"生成された検索クエリ: `{search_query}`") # デバッグ用
        return search_query
    except Exception as e:
        st.warning(f"検索クエリの生成に失敗しました: {e}。元のプロンプトを使用します。")
        return prompt


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
    # 非同期処理の初期化問題を解決するため、全ての処理の前にイベントループを設定
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError("API key not found")
    genai.configure(api_key=api_key)
    
    # 埋め込みモデルと生成モデルを初期化
    model = genai.GenerativeModel("gemini-2.5-pro")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

except KeyError:
    st.error("Gemini APIキーが設定されていません。Streamlit Cloudの設定でGEMINI_API_KEYまたはGOOGLE_API_KEYを追加してください。")
    st.stop()

# --- 知識ベース構築 ---
KNOWLEDGE_SOURCES = [
    ("オーダーノート現実創造プログラム", "**/*.txt"),
]
FAISS_INDEX_PATH = "data/faiss_index_v3" # インデックスを強制再構築するためパスを変更

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
        
        # 発見したファイルリストをデバッグ用に表示
        with st.expander(f"発見された {len(all_file_paths)} 件のファイルリスト（最初の30件）"):
            st.code('\n'.join(sorted(all_file_paths)[:30]))

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

# --- サイドバーにナレッジソース情報を表示 ---
st.sidebar.title("⚙️ ナレッジベース情報")
try:
    source_directory, pattern = KNOWLEDGE_SOURCES[0]
    search_path = os.path.join(source_directory, pattern)
    all_file_paths = sorted(glob.glob(search_path, recursive=True))

    st.sidebar.success(f"**読み込み元フォルダ:**\\n`{source_directory}`")

    with st.sidebar.expander(f"**読み込み対象ファイル: {len(all_file_paths)}件**"):
        st.markdown("---")
        # パスを整形して表示
        display_text = "\\n".join([f"- `{os.path.relpath(p)}`" for p in all_file_paths])
        st.markdown(display_text)

except Exception as e:
    st.sidebar.error(f"ファイルリストの取得中にエラー: {e}")


db = load_faiss_index(FAISS_INDEX_PATH, embeddings)

# --- AIによる検索クエリ生成 ---
def generate_search_query(prompt, conversation_history):
    """ユーザーのプロンプトと会話履歴から、FAISS検索に最適なクエリを生成する"""
    
    # 会話履歴を整形
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    # プロンプトテンプレート
    prompt_template = f"""
あなたは優秀な検索アシスタントです。以下の会話履歴と最後のユーザープロンプトを分析し、ベクトルデータベースから最も関連性の高い情報を引き出すための、簡潔かつ効果的な検索クエリを生成してください。

【制約条件】
- 最も重要なキーワードを抽出してください。
- 検索クエリ以外の余計な文章は含めないでください。
- ユーザーの意図を正確に反映してください。

【会話履歴】
{history_str}

【最後のユーザープロンプト】
{prompt}

【生成すべき検索クエリ】
"""
    
    try:
        # Geminiに検索クエリの生成を依頼
        response = model.generate_content(prompt_template)
        search_query = response.text.strip()
        # st.info(f"生成された検索クエリ: `{search_query}`") # デバッグ用
        return search_query
    except Exception as e:
        st.warning(f"検索クエリの生成に失敗しました: {e}。元のプロンプトを使用します。")
        return prompt


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
                    st.markdown(f"**選択理由:** {reason}")
                    
                    # コンテナを使って参照箇所をスクロール可能に
                    content_html = f"""
                        <div style="background-color: #262730; border-radius: 5px; padding: 10px; height: 150px; overflow-y: auto; border: 1px solid #333;">
                            <pre style="white-space: pre-wrap; word-wrap: break-word; color: #FAFAFA; font-family: 'Source Code Pro', monospace;">{doc.page_content}</pre>
                        </div>
                    """
                    components.html(content_html, height=170)
                    st.divider()

# --- ユーザーからの入力 ---
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    
    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="assets/avatar.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. 最適な検索クエリを生成
            with st.spinner("最適な検索方法を考えています..."):
                 search_query = generate_search_query(prompt, st.session_state.messages)

            # 2. 生成されたクエリでFAISSを検索
            with st.spinner("ナレッジベースを検索しています..."):
                docs_with_scores = db.similarity_search_with_score(search_query, k=10)
            
            # 3. 検索結果を整形
            source_texts = []
            source_docs_with_reasons = []
            is_relevant_info_found = False

            if docs_with_scores:
                for doc, score in docs_with_scores:
                    # ここですでに関連性があるかどうかの大まかな判断を入れる
                    # 例: スコアが一定以上、または特定のキーワードを含むなど
                    # 今回はシンプルに上位5件は関連ありとみなす
                    if len(source_docs_with_reasons) < 5:
                         source_docs_with_reasons.append({
                            "doc": doc,
                            "reason": f"AIの判断による関連候補 (スコア: {score:.4f})"
                        })
                         source_texts.append(doc.page_content)
                         is_relevant_info_found = True

                # 関連するものが1つでもあれば、コンテキストを構築
                if is_relevant_info_found:
                    context = "--- 関連情報 ---\n"
                    # コンテキストに含める情報を上位5件に制限
                    for item in source_docs_with_reasons:
                        context += item["doc"].page_content + "\n\n"
                else:
                    context = "--- 関連情報 ---\nなし"

            # --- システムプロンプトの準備 ---
            with open("引き寄せアプリ/docs/system_prompt.md", "r", encoding="utf-8") as f:
                final_system_prompt = f.read()

            final_system_prompt += f"\n\n{context}"
            
            # --- API呼び出しとストリーミング ---
            history = []
            for m in st.session_state.messages[:-1]:
                role = 'user' if m['role'] == 'user' else 'model'
                history.append({'role': role, 'parts': [{'text': m['content']}]})

            chat = model.start_chat(history=history)
            
            prompt_with_context = f"{final_system_prompt}\n\nuser: {prompt}\nassistant:"

            # ストリーミングを有効にして応答を生成
            stream = chat.send_message(
                prompt,
                stream=True,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                )
            )

            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            
            # 最終的な応答をクリーンアップして表示
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"応答の生成中にエラーが発生しました: {e}")
            full_response = "申し訳ありません、応答を生成できませんでした。"
            message_placeholder.markdown(full_response)

        # 応答と参照元をメッセージ履歴に追加
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