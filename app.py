import os
import json
import tempfile
import urllib.parse
from pathlib import Path
from flask import Flask, request, jsonify

# External packages
import requests
import trafilatura
import gspread
import bs4
from google.oauth2.service_account import Credentials
from openai import OpenAI
from google.cloud import texttospeech
from moviepy.editor import *
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuthCreds

# ------------------------------------------------------------
# ロギング設定
# ------------------------------------------------------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

# ----------------------------------------------------------------------------
# OpenAI コンテンツモデレーション補助関数
# ----------------------------------------------------------------------------

def log_moderation(text: str, context: str = ""):
    """OpenAI Moderation API で NG カテゴリを検出しログに残す。失敗しても処理を止めない。"""
    if DISABLE_NARRATION:
        return
    try:
        mod = client.moderations.create(model="omni-moderation-latest", input=text[:5000])
        result = mod.results[0]
        logging.info("moderation (%s): flagged=%s, categories=%s", context, result.flagged, result.categories)
    except Exception:
        logging.exception("moderation (%s): API call failed", context)


# ----------------------------------------------------------------------------
# 環境変数 & 定数
# ----------------------------------------------------------------------------
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
SHEET_URL = os.environ.get("SHEET_URL")  # 例: https://docs.google.com/...
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")  # 共有ドライブ内のフォルダID
# Secret Manager で以下を注入: GCP_SA_JSON_SECRET, YOUTUBE_TOKEN_JSON
DISABLE_IMAGE_GEN = os.environ.get("DISABLE_IMAGE_GEN") == "1"
DISABLE_NARRATION = os.environ.get("DISABLE_NARRATION") == "1"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/cloud-platform",
]

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------------------------
# 認証ヘルパ
# ----------------------------------------------------------------------------

def get_service_account_creds():
    """Secret Manager から取得したサービスアカウントJSONをCredentialsへ"""
    sa_json = os.environ.get("GCP_SA_JSON_SECRET")
    if not sa_json:
        raise RuntimeError("環境変数 GCP_SA_JSON_SECRET が未設定です")
    info = json.loads(sa_json)
    return Credentials.from_service_account_info(info, scopes=SCOPES)


def get_youtube_creds():
    """token.json 相当を環境変数からロードし、必要なら refresh"""
    token_json = os.environ.get("YOUTUBE_TOKEN_JSON")
    if not token_json:
        raise RuntimeError("環境変数 YOUTUBE_TOKEN_JSON が未設定です")
    token_path = "/tmp/token.json"
    with open(token_path, "w", encoding="utf-8") as fp:
        fp.write(token_json)
    creds = OAuthCreds.from_authorized_user_file(token_path)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

# ----------------------------------------------------------------------------
# Google Sheets Access
# ----------------------------------------------------------------------------

def get_sheet():
    creds = get_service_account_creds()
    gc = gspread.authorize(creds)
    url = SHEET_URL.split("#")[0].split("?")[0] if SHEET_URL else ""
    try:
        return gc.open_by_url(url).sheet1
    except gspread.exceptions.NoValidUrlKeyFound:
        # fallback: extract key manually
        import re
        m = re.search(r"/d/([\w-]+)", url)
        if not m:
            raise
        return gc.open_by_key(m.group(1)).sheet1

# ----------------------------------------------------------------------------
# 1) 記事取得
# ----------------------------------------------------------------------------

def fetch_article(url: str) -> str:
    text_parts, visited = [], set()
    logging.info(f"fetch_article: start url={url}")
    while url and url not in visited:
        logging.info("fetch_article: visiting %s", url)
        visited.add(url)
        resp = requests.get(url, timeout=15)
        article_text = trafilatura.extract(resp.text, favor_recall=True)
        if article_text:
            text_parts.append(article_text)
        soup = bs4.BeautifulSoup(resp.text, "lxml")
        next_link = soup.find("link", rel="next")
        url = urllib.parse.urljoin(url, next_link["href"]) if next_link else None
    return "\n\n".join(text_parts).strip()

# ----------------------------------------------------------------------------
# 2) 画像生成 (GPT-4o + DALL·E3)
# ----------------------------------------------------------------------------

def generate_images(article: str, n: int = 3):
    # 画像生成無効化フラグ
    if DISABLE_IMAGE_GEN:
        logging.info("generate_images: disabled via env var, skipping image generation")
        return []

    # 1) シーン説明文を取得 --------------------------------------------------
    prompt = (
        "以下の記事内容を読んで、ニュース動画用の背景シーン説明を" f"{n}個、日本語で1行ずつ出力してください。"
    )
    logging.info("generate_images: prompt=%s", prompt)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt + "\n\n" + article[:4000]}],
        )
    except Exception as e:
        logging.exception("generate_images: scene generation failed")
        raise

    scenes = [l.strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    logging.info("generate_images: scenes=%s", scenes)
    # 各シーンと記事本文のモデレーション結果を記録
    log_moderation(article[:8000], "article")
    for s in scenes:
        log_moderation(s, "scene")

    # 2) 各シーンを画像化 ------------------------------------------------------
    paths = []
    for i, scene in enumerate(scenes[:n]):
        retries = 0
        while retries < 3:
            try:
                logging.info("generate_images: request image scene=%s", scene)
                img_resp = client.images.generate(
                    model="dall-e-3",
                    prompt=scene,
                    size="1024x1024",
                )
                url = img_resp.data[0].url
                logging.info("generate_images: image url=%s", url)
                raw = requests.get(url, timeout=30).content
                path = f"/tmp/scene_{i}.png"
                with open(path, "wb") as fp:
                    fp.write(raw)
                paths.append(path)
                break  # success
            except Exception as e:
                retries += 1
                logging.exception(
                    "generate_images: image generation failed (attempt %d/3) scene=%s", retries, scene
                )
                if retries == 3:
                    logging.error("generate_images: giving up on scene %s after 3 retries", scene)
        
    if not paths:
        raise RuntimeError("generate_images: failed to generate any images")
    return paths

# ----------------------------------------------------------------------------
# 3) 読み上げ原稿生成
# ----------------------------------------------------------------------------

def create_narration(article: str) -> str:
    if DISABLE_NARRATION:
        logging.error("create_narration: disabled via env, aborting")
        raise RuntimeError("ナレーション生成無効 (DISABLE_NARRATION=1)")
    prompt = (
        "次のニュース記事本文を、自然な日本語の朗読原稿に整形してください。\n"
        "・見出し行の後に改行で間を取る\n"
        "・不自然な表現は口語に言い換える\n"
        "・全体で3分以内に収まるよう要約も可\n\n記事:\n" + article[:8000]
    )
    try:
        res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    except Exception:
        logging.exception("create_narration: OpenAI chat completion failed for article snippet=%s", article[:500])
        raise
    script = res.choices[0].message.content.strip()
    if not script:
        raise RuntimeError("create_narration: empty script from OpenAI")
    return script

# ----------------------------------------------------------------------------
# 4) 朗読原稿処理 & 音声合成 (Google TTS)
# ----------------------------------------------------------------------------

def prepare_tts_script(raw_text: str) -> str:
    """TTS 用にテキストを SSML に整形する。

    ・見出しの前後に長めのポーズ (<break>) を挿入
    ・記号や URL は読み飛ばす
    ・連続空白を整理
    ・辞書で指定した漢字をふりがなへ置換
    """
    import re, html

    # 誤読しやすい漢字 → よみがな 辞書
    KANJI_READING: dict[str, str] = {
        "刷新": "さっしん",
        "行使": "こうし",
        "一段": "いちだん",
    }

    text = raw_text
    # Markdown 見出しやリスト記号を削除
    text = re.sub(r"^[#*]+\\s*", "", text, flags=re.M)
    # URL を削除
    text = re.sub(r"https?://\S+", "", text)
    # 余分な空白を縮約
    text = re.sub(r"[ \t]+", " ", text)

    # 辞書で漢字を置換
    if KANJI_READING:
        pattern = re.compile("|".join(map(re.escape, KANJI_READING.keys())))
        text = pattern.sub(lambda m: KANJI_READING[m.group(0)], text)

    # 行単位で SSML
    ssml_lines: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # 見出し判定: 30文字以内かつ記号少なめ
        if len(line) <= 30 and not re.search(r"[。.,]", line):
            ssml_lines.append(f"<break time=\"700ms\"/>{html.escape(line)}<break time=\"700ms\"/>")
        else:
            ssml_lines.append(f"{html.escape(line)}<break time=\"300ms\"/>")

    return f"<speak>{''.join(ssml_lines)}</speak>"

def synthesize_speech(text: str) -> str:
    """読み上げ用原稿を SSML へ変換して Google TTS で合成。"""
    ssml = prepare_tts_script(text)

    creds = get_service_account_creds()
    tts = texttospeech.TextToSpeechClient(credentials=creds)
    input_text = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ja-JP", name="ja-JP-Standard-B"
    )
    audio_cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.05,
    )
    try:
        audio = tts.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_cfg,
        )
    except Exception:
        logging.exception("synthesize_speech: Google TTS failed for text snippet=%s", text[:200])
        raise
    out_path = "/tmp/voice.mp3"
    with open(out_path, "wb") as fp:
        fp.write(audio.audio_content)
    return out_path

# ----------------------------------------------------------------------------
# 5) 動画生成 (moviepy)
# ----------------------------------------------------------------------------

def make_video(images: list[str], voice_path: str) -> str:
    audio_clip = AudioFileClip(voice_path)
    duration = audio_clip.duration
    if images:
        img_duration = duration / len(images)
        clips = [ImageClip(p).set_duration(img_duration) for p in images]
    else:
        # 黒背景のみで動画を生成
        from moviepy.editor import ColorClip
        clips = [ColorClip(size=(1280, 720), color=(0, 0, 0)).set_duration(duration)]
    video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
    out_path = "/tmp/news_video.mp4"
    video.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=30)
    return out_path

# ----------------------------------------------------------------------------
# 6) Google Drive アップロード
# ----------------------------------------------------------------------------

def upload_drive(file_path: str) -> str:
    """Upload video to Google Drive folder and return share URL."""
    creds = get_service_account_creds()
    drive = build("drive", "v3", credentials=creds)

    # フォルダ存在確認（shared drive も考慮して supportsAllDrives=True）
    try:
        drive.files().get(fileId=DRIVE_FOLDER_ID, fields="id", supportsAllDrives=True).execute()
    except Exception as e:
        logging.error("upload_drive: folder id %s not accessible: %s", DRIVE_FOLDER_ID, e)
        raise

    meta = {"name": Path(file_path).name, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    file = drive.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    # 公開リンク作成
    drive.permissions().create(fileId=file["id"], body={"role": "reader", "type": "anyone"}).execute()
    return f"https://drive.google.com/file/d/{file['id']}/view"

# ----------------------------------------------------------------------------
# 7) YouTube アップロード
# ----------------------------------------------------------------------------

def upload_youtube(file_path: str, title: str) -> str:
    yt_creds = get_youtube_creds()
    yt = build("youtube", "v3", credentials=yt_creds)
    body = {
        "snippet": {
            "title": title,
            "description": title,
            "categoryId": "25",
        },
        "status": {"privacyStatus": "public"},
    }
    media = MediaFileUpload(file_path, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    return f"https://youtu.be/{resp['id']}"

# ----------------------------------------------------------------------------
# 8) スプレッドシート更新
# ----------------------------------------------------------------------------

def update_sheet(row: int, drive_url: str, yt_url: str):
    sheet = get_sheet()
    if drive_url:
        sheet.update(f"B{row}", drive_url)
    if yt_url:
        sheet.update(f"C{row}", yt_url)

# ----------------------------------------------------------------------------
# メイン処理関数
# ----------------------------------------------------------------------------

def process_row(row: int, url: str):
    sheet = get_sheet()
    article = fetch_article(url)

    # 画像生成
    images: list[str] = []
    try:
        images = generate_images(article)
    except Exception as e:
        sheet.update(f"D{row}", f"画像生成失敗: {e}")
        raise

    # ナレーション生成
    script = ""
    try:
        script = create_narration(article)
    except Exception as e:
        sheet.update(f"D{row}", f"原稿生成失敗: {e}")
        raise

    # 判定: 画像または原稿が無い場合はスキップ扱い
    if not images or not script.strip():
        reason = []
        if not images:
            reason.append("画像なし")
        if not script.strip():
            reason.append("原稿なし")
        sheet.update(f"D{row}", "/".join(reason) or "スキップ")
        logging.info("process_row: row %d skipped (%s)", row, reason)
        return {"row": row, "skipped": True, "reason": reason}

    # 音声合成
    voice = synthesize_speech(script)
    # 動画生成
    video = make_video(images, voice)
    # アップロード
    drive_url = upload_drive(video)
    yt_url = upload_youtube(video, script.split("\n")[0][:50])
    update_sheet(row, drive_url, yt_url)
    return {"row": row, "drive": drive_url, "yt": yt_url}

# ----------------------------------------------------------------------------
# Flask ルーティング
# ----------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    logging.info("/process: start processing spreadsheet rows")
    results = []
    sheet = get_sheet()
    rows = sheet.get_all_values()[1:]  # ヘッダー除外
    for idx, r in enumerate(rows, start=2):
        url = r[0] if len(r) else ""
        done = r[1] if len(r) > 1 else ""
        if url and not done:
            try:
                res = process_row(idx, url)
                res["status"] = "success"
                results.append(res)
            except Exception as e:
                logging.exception("process_row failed")
                sheet.update(f"D{idx}", str(e))
                results.append({"row": idx, "error": str(e)})
    return jsonify(results)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # ローカルテスト用
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
