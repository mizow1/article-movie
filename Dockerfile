# ベースイメージ
FROM python:3.11-slim

# moviepy が必要とする ffmpeg と X ライブラリを追加
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# ポート
EXPOSE 8080

# 実行コマンド
CMD ["python", "app.py"]
