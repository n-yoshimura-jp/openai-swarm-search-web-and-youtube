# ライブラリをインポート
import os
from getpass import getpass
from swarm import Swarm, Agent
from openai import OpenAI
from duckduckgo_search import DDGS
import pandas as pd
from googleapiclient.discovery import build

# OpenAIのAPIキーを入力
openai_api_key = getpass(prompt="Please enter OpenAI API key:")

# YouTube Data APIのAPIキーを入力
youtube_data_api_key = getpass(prompt="Please enter YouTube Data API key:")

# OpenAIのAPIキーを環境変数を設定
os.environ["OPENAI_API_KEY"] = openai_api_key

# Swarmクライアントを初期化
client = Swarm()

# Web検索を行う関数
def search_web(keyword: str) -> pd.DataFrame:
    try:
        # Web検索
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=keyword, 
                region="jp-jp", 
                safesearch="off", 
                timelimit=None, 
                max_results=10
            ))
        
        # データをデータフレームに変換
        data = [
            {
                "タイトル": result.get("title", ""),
                "URL": result.get("href", ""),
                "本文": result.get("body", "")
            } 
            for result in results
        ]
        df = pd.DataFrame(data)
        df.index = df.index + 1
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # エラー時には空のデータフレームを返す

# YouTube動画を検索する関数
def search_youtube_videos(query: str) -> pd.DataFrame:
    try:
        youtube = build("youtube", "v3", developerKey=youtube_data_api_key)

        # 動画検索
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=10
        )
        response = request.execute()

        # 結果をデータフレームに変換
        data = []
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            channel_title = item["snippet"]["channelTitle"]
            publish_time = item["snippet"]["publishedAt"]

            data.append({
                "タイトル": title,
                "説明": description,
                "チャンネル名": channel_title,
                "公開日時": publish_time,
                "URL": f"https://www.youtube.com/watch?v={video_id}"
            })

        df = pd.DataFrame(data)
        df.index = df.index + 1

        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # エラー時は空のデータフレームを返す

# トリアージエージェントへのインストラクション
triage_agent_instructions = """
あなたはユーザーの要求を適切に判定できるトリアージの専門家です。ユーザーの要求に応じて、適切なエージェントに転送してください。
YouTubeの検索が必要な場合は、youtube_search_agentに会話を転送してください。
Web検索が必要な場合は、web_search_agentに会話を転送してください。
転送が不要で、直接質問に回答できる場合は回答してください。
"""

# トリアージエージェント
triage_agent = Agent(
    name="トリアージエージェント",
    model="gpt-4o-mini",
    instructions=triage_agent_instructions
)

# Web検索エージェントへのインストラクション
web_search_agent_instructions = """
あなたはWeb検索の専門家です。ユーザーからキーワードでWeb検索を要求されたら、その結果をデータフレームで提供してください。検索については、search_webの関数を使用してください。
Web検索が不要な場合は、triage_agentに会話を転送してください。
YouTube検索が必要な場合は、youtube_search_agentに会話を転送してください。
"""

# Web検索エージェント
web_search_agent = Agent(
    name="Web検索エージェント",
    model="gpt-4o-mini",
    instructions=web_search_agent_instructions,
    functions=[search_web]
)

# YouTube検索エージェントへのインストラクション
youtube_search_agent_instructions = """
あなたはYouTube検索の専門家です。ユーザーからキーワードでYouTube検索を要求されたら、その結果をデータフレームで提供してください。検索については、search_youtube_videosの関数を使用してください。
YouTube検索が不要な場合は、triage_agentに会話を転送してください。
Web検索が必要な場合は、web_search_agentに会話を転送してください。
"""

# YouTube検索エージェント
youtube_search_agent = Agent(
    name="YouTube検索エージェント",
    model="gpt-4o-mini",
    instructions=youtube_search_agent_instructions,
    functions=[search_youtube_videos]
)

# Web検索エージェントに転送する関数
def transfer_to_web_search_agent():
    print("Web検索エージェントに転送します")
    return web_search_agent

# YouTube検索エージェントに転送する関数
def transfer_to_youtube_search_agent():
    print("YouTube検索エージェントに転送します")
    return youtube_search_agent

# トリアージエージェントに転送に転送する関数
def transfer_back_to_triage_agent():
    print("トリアージエージェントに転送します")
    return triage_agent

# triage_agentにWeb検索エージェント、YouTube検索エージェントに転送する関数を指定
triage_agent.functions = [transfer_to_web_search_agent, transfer_to_youtube_search_agent]

# web_search_agentにトリアージエージェントに転送に転送する関数を追加
web_search_agent.functions.append(transfer_back_to_triage_agent)

# web_search_agentにYouTube検索エージェントに転送に転送する関数を追加
web_search_agent.functions.append(transfer_to_youtube_search_agent)

# youtube_search_agentにトリアージエージェントに転送に転送する関数を追加
youtube_search_agent.functions.append(transfer_back_to_triage_agent)

# youtube_search_agentにWeb検索エージェントに転送に転送する関数を追加
youtube_search_agent.functions.append(transfer_to_web_search_agent)

# エージェントを呼び出して、実行 (Web検索)
response_web_search = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "宮崎グルメというキーワードでWeb検索をしてください。"}]
)

# 結果を確認
print(response_web_search.messages[-1]["content"])

# エージェントを呼び出して、実行 (Web検索)
response_youtube_search = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "宮崎グルメというキーワードでYouTube検索をしてください。"}]
)

# 結果を確認
print(response_youtube_search.messages[-1]["content"])

# エージェントを呼び出して、実行 (通常の質問)
response_triage_agent = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "宮崎グルメでおすすめを教えてください。"}]
)

# 結果を確認
print(response_triage_agent.messages[-1]["content"])