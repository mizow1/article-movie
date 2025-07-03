# Article Movie Generator

ニュース記事を自動で動画コンテンツに変換し、複数のプラットフォームに同時配信するFlaskアプリケーションです。

## 概要

このアプリケーションは以下の機能を提供します：

- ニュース記事のウェブスクレイピングと内容抽出
- OpenAI GPT-4oを使用した記事要約と画像生成
- Google Text-to-Speechによる音声合成
- MoviePyを使用した動画生成（横長・縦長の2形式）
- 複数プラットフォームへの自動投稿
  - YouTube（通常動画・Shorts）
  - Instagram Reels
  - TikTok
  - Twitter/X
- Google Sheetsとの連携による処理管理

## 主要機能

### 1. 記事処理
- URLから記事タイトルと本文を自動抽出
- OpenAI Moderation APIによるコンテンツフィルタリング
- 記事内容の要約と読み上げ原稿の生成

### 2. コンテンツ生成
- DALL-E 3による記事に関連した画像生成
- Google TTSによる日本語音声合成（SSML対応）
- 横長（16:9）と縦長（9:16）の動画形式に対応
- SRT字幕ファイルの自動生成

### 3. 配信プラットフォーム
- **YouTube**: 通常動画とShortsの両方に対応、字幕付き
- **Instagram**: Reels形式での投稿
- **TikTok**: 縦長動画での投稿  
- **Twitter/X**: 動画付きツイート

### 4. 処理管理
- Google Sheets記載のURLの記事を元に動画を生成し、複数プラットフォームに同時配信する
- エラーハンドリングとログ記録

## 技術スタック

- **Backend**: Flask (Python)
- **AI/ML**: OpenAI GPT-4o, DALL-E 3, OpenAI Moderation API
- **TTS**: Google Cloud Text-to-Speech
- **動画処理**: MoviePy
- **ウェブスクレイピング**: trafilatura, BeautifulSoup
- **クラウドサービス**: Google Sheets API, Google Drive API, YouTube Data API
- **認証**: Google OAuth2, サービスアカウント

## 環境変数

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google Services
GCP_SA_JSON_SECRET=your_service_account_json
YOUTUBE_TOKEN_JSON=your_youtube_oauth_token
SHEET_URL=your_google_sheets_url
DRIVE_FOLDER_ID=your_drive_folder_id

# Social Media APIs (オプション)
IG_ACCESS_TOKEN=your_instagram_token
IG_USER_ID=your_instagram_user_id
TIKTOK_ACCESS_TOKEN=your_tiktok_token
TIKTOK_CLIENT_KEY=your_tiktok_client_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret

# 機能制御フラグ
DISABLE_IMAGE_GEN=0  # 1で画像生成無効
DISABLE_NARRATION=0  # 1で音声合成無効

# サーバー設定
PORT=8080
```

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法
### 1. 動画の元記事をGoogle Sheetsに記載
- Google Sheetsに記事URLと処理フラグ「1」を設定。
    - [https://x.gd/acQpT](https://x.gd/acQpT)

### 2. 処理実行
- Google Cloud Runへデプロイ
    ```bash
    gcloud config set project article-movie
    SHEET="https://docs.google.com/spreadsheets/d/1GJJlgi661ofOfDHqjVIvdDEa9xmhK2SHyyxiqHi3RuA/edit?gid=0" && gcloud run deploy news-video-generator --source=. --region=asia-northeast1 --project=article-movie --memory=2Gi --cpu=2 --timeout=3600 --set-env-vars=SHEET_URL=${SHEET},DRIVE_FOLDER_ID=1DQWj1mSSFwjf02odaZDO3_2QNn7wr58Q,DISABLE_IMAGE_GEN=1 --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest,GCP_SA_JSON_SECRET=GCP_SA_JSON_SECRET:latest,YOUTUBE_TOKEN_JSON=YOUTUBE_TOKEN_JSON:latest
    ```
- 出力
    ```bash
    curl -X POST https://news-video-generator-907276727531.asia-northeast1.run.app/process
    ```
- 成功すればスプレッドシートに動画URLが出力される。
- 失敗時はログで原因追跡
    ```bash
    gcloud run services logs read news-video-generator --limit=100 --region=asia-northeast1
    ```

### 3. Google Sheetsの形式
| A列(URL) | B列(動画URL) | C列(YouTube) | D列(エラー内容) | E列(実行フラグ) | F列(公開日) | G列(表記原稿) | H列(読み上げ原稿) | I列(YT Shorts) | J列(Instagram) | K列(TikTok) | L列(Twitter) |
|----------|------------|--------------|-------------|-----------|-----------|--------------|-------------|----------------|----------------|-------------|--------------|
| 記事URL | 動画URL | YouTube URL | エラー内容 | 実行フラグ | 公開日 | 表記原稿 | 読み上げ原稿 | Shorts URL | Instagram URL | TikTok URL | Twitter URL |

## Docker対応

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
```

## 注意事項

- 動画生成には下記コストが必要。（エラーで生成されない時は不要）
- 文章：動画1分あたり5円
- 音声：動画1分あたり5円
- 画像：1画像20円（動画30秒ごとに1枚作成）

## ライセンス

このプロジェクトは個人/商用利用可能ですが、各APIプロバイダーの利用規約に従ってください。