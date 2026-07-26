import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


def main():
    load_dotenv()

    api_key = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY"),
    )
    model_name = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    )

    if not api_key:
        st.error(
            "GEMINI_API_KEYが設定されていません。"
            ".envファイルを確認してください。"
        )
        st.stop()

    # Geminiクライアントを作成
    client = genai.Client(api_key=api_key)

    # ページ設定
    st.set_page_config(
        page_title="AIアシスタント",
        page_icon="🤖",
    )

    st.title("AIアシスタントへようこそ。")
    st.caption(f"使用モデル: {model_name}")

    # Streamlit上で保持する会話履歴
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 過去の会話を画面に表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力
    if user_input := st.chat_input(
        "AIアシスタントにメッセージを入力"
    ):
        # ユーザーメッセージを履歴に追加
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # ユーザーメッセージをすぐ表示
        with st.chat_message("user"):
            st.markdown(user_input)

        # Streamlitの履歴をGemini形式へ変換
        gemini_history = []

        for message in st.session_state.messages:
            gemini_role = (
                "model"
                if message["role"] == "assistant"
                else "user"
            )

            gemini_history.append(
                types.Content(
                    role=gemini_role,
                    parts=[
                        types.Part.from_text(
                            text=message["content"]
                        )
                    ],
                )
            )

        # Geminiからストリーミング応答を取得
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                response_stream = (
                    client.models.generate_content_stream(
                        model=model_name,
                        contents=gemini_history,
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "あなたは親切で丁寧な"
                                "AIアシスタントです。"
                            )
                        ),
                    )
                )

                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(
                            full_response + "▌"
                        )
                        time.sleep(0.04)

                placeholder.markdown(full_response)

            except Exception as error:
                st.error(
                    "Gemini APIへの接続中に"
                    "エラーが発生しました。"
                )
                st.exception(error)
                return

        # AIの回答を会話履歴に保存
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )


if __name__ == "__main__":
    main()