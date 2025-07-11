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

def fetch_article(url: str) -> tuple[str, str]:  # returns (title, body)
    text_parts, visited = [], set()
    logging.info(f"fetch_article: start url={url}")
    title = ""
    while url and url not in visited:
        logging.info("fetch_article: visiting %s", url)
        visited.add(url)
        resp = requests.get(url, timeout=15)
        html_bytes = resp.content  # raw bytes to avoid premature decoding
        if not title:
            # og:title or <title>
            try:
                soup_ = bs4.BeautifulSoup(html_bytes, "lxml", from_encoding=None)
                og = soup_.find("meta", property="og:title")
                title = og["content"].strip() if og and og.get("content") else ""
                if not title and soup_.title:
                    title = soup_.title.get_text(strip=True)
            except Exception:
                pass
        try:
            article_text = trafilatura.extract(html_bytes, favor_recall=True)
        except Exception:
            article_text = trafilatura.extract(resp.text, favor_recall=True)
        if article_text:
            text_parts.append(article_text)
        soup = bs4.BeautifulSoup(html_bytes, "lxml", from_encoding=None)
        next_link = soup.find("link", rel="next")
        url = urllib.parse.urljoin(url, next_link["href"]) if next_link else None
    body = "\n\n".join(text_parts).strip()
    return title or "", body

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
# ナレーション後処理: 記号除去・ふりがな簡略化
# ----------------------------------------------------------------------------

def clean_narration(text: str) -> str:
    """字幕/音声向けに不要なマークダウン記号や重複語を除去する。

    処理内容:
    1. 行頭の Markdown 見出し/箇条書き記号 (#, ##, -, *) を除去。
    2.  "語（よみ）" 形式は読みのみ残す。
    3. 重複スペースを1つへ。
    """
    import re

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        # 行頭 Markdown 記号除去
        # **bold** を除去
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        # 行頭 Markdown 記号除去
        line = re.sub(r"^\s*[#>\-*]+\s*", "", line)
        # 読み優先:  漢字（かな）→かな
        def _kana_sub(match):
            kanji, reading = match.group(1), match.group(2)
            # ひらがなorカタカナのみなら読みを採用、そうでなければ元
            return reading if re.fullmatch(r"[ぁ-ゖァ-ヺー]+", reading) else reading
        line = re.sub(r"([^\s（）()]+)[（(]([^）)]+)[）)]", _kana_sub, line)
        cleaned_lines.append(line.strip())
    text = "\n".join(l for l in cleaned_lines if l)
    # 連続空白
    text = re.sub(r"\s{2,}", " ", text)
    return text

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
    # Markdown 記号 (行頭/行中の #,*,-,数字. など) を削除
    text = re.sub(r"^[#*\-\d\.]+\s*", "", text, flags=re.M)  # 行頭
    text = re.sub(r"\s[#*]{1,6}\s*", " ", text)  # 行中
    # URL を削除
    text = re.sub(r"https?://\S+", "", text)
    # 余分な空白を縮約 (全角スペースは保持し、後でポーズへ変換)
    text = re.sub(r"[ \t]+", " ", text)

    # 辞書で漢字を置換
    if KANJI_READING:
        pattern = re.compile("|".join(map(re.escape, KANJI_READING.keys())))
        text = pattern.sub(lambda m: KANJI_READING[m.group(0)], text)

    # 行単位で SSML
    ssml_lines: list[str] = []
    # 行内の全角スペースをポーズへ変換（Escape 前にプレースホルダーに置換し、あとで SSML タグへ戻す）
    BREAK_TOKEN = "__BR300__"
    text = text.replace("　", BREAK_TOKEN)

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # 見出し判定: 30文字以内かつ記号少なめ
        if len(line) <= 30 and not re.search(r"[。.,]", line):
            escaped = html.escape(line).replace(BREAK_TOKEN, '<break time="300ms"/>')
            ssml_lines.append(f"<break time=\"700ms\"/>{escaped}<break time=\"700ms\"/>")
        else:
            escaped = html.escape(line).replace(BREAK_TOKEN, '<break time="300ms"/>')
            ssml_lines.append(f"{escaped}<break time=\"300ms\"/>")

    return f"<speak>{''.join(ssml_lines)}</speak>"

# ----------------------------------------------------------------------------
# 字幕分割 & SRT 生成
# ----------------------------------------------------------------------------

def _split_long_sentence(sentence: str, max_len: int) -> list[str]:
    """句点で分割した後、さらに長い文は読点や単語単位で分割"""
    import re
    if len(sentence) <= max_len:
        return [sentence]
    segments: list[str] = []
    parts = sentence.split("、")  # 読点で分割
    buf = ""
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        add = ("、" if i < len(parts) - 1 else "") + part
        if len(buf) + len(add) > max_len and buf:
            segments.append(buf + ("。" if not buf.endswith("。") else ""))
            buf = part + ("、" if i < len(parts) - 1 else "")
        else:
            buf += add
    if buf:
        if not buf.endswith("。"):
            buf += "。"
        segments.append(buf)
    # fallback: still too long, brute split
    result: list[str] = []
    for seg in segments:
        while len(seg) > max_len:
            result.append(seg[:max_len])
            seg = seg[max_len:]
        if seg:
            result.append(seg)
    return result or [sentence]

def segment_subtitles(text: str, max_len: int = 50) -> list[str]:
    """原稿テキストを字幕セグメントへ分割"""
    import re
    sentences = re.split(r"(?<=。)\s*", text)
    segments: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:
            segments.extend(_split_long_sentence(s, max_len))
        else:
            segments.append(s)
    return segments

def _sec_to_ts(sec: float) -> str:
    hrs = int(sec // 3600)
    mins = int((sec % 3600) // 60)
    secs = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

def _align_segments_to_timepoints(segments: list[str], timepoints: list) -> list[tuple[float, float]]:
    """単語タイムスタンプを用いて各セグメントの開始・終了秒を返す"""
    # timepoints: list of google.cloud.texttospeech.Timepoint
    if not timepoints:
        return []
    # 時系列ソート
    tp = sorted(timepoints, key=lambda x: x.time_seconds)
    # word list from script
    words = []
    import re
    for s in segments:
        words.extend(re.findall(r"\w+", s))
    if len(words) == 0 or len(tp) < len(words):
        return []
    # Map each word to timepoint index (assume 1-1)
    seg_times: list[tuple[float, float]] = []
    w_idx = 0
    for s in segments:
        w_cnt = len(re.findall(r"\w+", s))
        start = tp[w_idx].time_seconds
        end = tp[min(w_idx + w_cnt - 1, len(tp)-1)].time_seconds
        seg_times.append((start, end))
        w_idx += w_cnt
    return seg_times

def generate_srt_file(script: str, duration: float, out_path: str = "/tmp/captions.srt", timepoints: list | None = None) -> str:
    """原稿と総尺から単純均等割りでSRTファイルを生成"""
    segments = segment_subtitles(script)
    if not segments:
        return ""
    # 正確なタイムスタンプがあれば使用
    seg_times: list[tuple[float, float]] = []
    if timepoints:
        seg_times = _align_segments_to_timepoints(segments, timepoints)
    if seg_times and len(seg_times) == len(segments):
        mode = "tp"
    else:
        mode = "even"
    total_chars = sum(len(s) for s in segments)
    if total_chars == 0 or duration <= 0:
        # Duration 未取得の場合は帰る
        return ""
    cur = 0.0
    with open(out_path, "w", encoding="utf-8") as fp:
        for idx, seg in enumerate(segments, 1):
            if mode == "tp" and seg_times:
                start, end = seg_times[idx-1]
            else:
                seg_len = len(seg)
                seg_dur = duration * seg_len / total_chars
                start = cur
                end = cur + seg_dur
            fp.write(f"{idx}\n")
            fp.write(f"{_sec_to_ts(start)} --> {_sec_to_ts(end)}\n")
            fp.write(f"{seg}\n\n")
            cur = end
    return out_path

def synthesize_speech(text: str, with_timepoints: bool = False):
    """読み上げ用原稿を SSML へ変換して Google TTS で合成。"""
    ssml = prepare_tts_script(text)
    # TimepointType はライブラリバージョンで場所が異なるため探索する
    enable_tp = None
    if with_timepoints:
        TimepointType = None
        # 1) 最新版 (google.cloud.texttospeech.TimepointType)
        TimepointType = getattr(texttospeech, "TimepointType", None)
        if TimepointType is None:
            # 2) v1 ラッパ (google.cloud.texttospeech_v1.types.SynthesizeSpeechRequest.TimepointType)
            try:
                from google.cloud import texttospeech_v1  # type: ignore
                TimepointType = texttospeech_v1.types.SynthesizeSpeechRequest.TimepointType  # type: ignore
            except Exception:
                TimepointType = None
        if TimepointType is None:
            logging.warning("synthesize_speech: TimepointType enum not found, proceeding without word timepoints")
        else:
            enable_tp = [TimepointType.WORD]

    creds = get_service_account_creds()
    tts = texttospeech.TextToSpeechClient(credentials=creds)
    input_text = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ja-JP", name="ja-JP-Standard-B"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )

    request_base = {
        "input": input_text,
        "voice": voice,
        "audio_config": audio_config,
    }
    if enable_tp:
        request_base["enable_time_pointing"] = enable_tp

    try:
        audio = tts.synthesize_speech(request=request_base)
    except TypeError as te:
        # 古いライブラリでは enable_time_pointing が未対応
        if "enable_time_pointing" in str(te):
            logging.warning("synthesize_speech: enable_time_pointing not supported, retrying without word timepoints")
            request_base.pop("enable_time_pointing", None)
            audio = tts.synthesize_speech(request=request_base)
            enable_tp = None  # タイムポイント取得なし
        else:
            raise
    except Exception as e:
        logging.error("synthesize_speech: Google TTS failed for text snippet=%s\n%s", text[:120], e)
        raise
    out_path = "/tmp/voice.mp3"
    with open(out_path, "wb") as fp:
        fp.write(audio.audio_content)
    if with_timepoints:
        return out_path, list(getattr(audio, "timepoints", []))
    return out_path

# ----------------------------------------------------------------------------
# 5) 動画生成 (moviepy)
# ----------------------------------------------------------------------------

def _sanitize_filename(s: str) -> str:
    """ファイル名に使えない文字を '_' に置換しトリム"""
    import re
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    return s.strip("_")

def make_video(
    images: list[str],
    voice_path: str,
    out_path: str,
    size: tuple[int, int] = (1280, 720),
    max_duration: int | None = None,
    fps: int = 30,
) -> str:
    """画像＋音声から単純なスライドショー動画を生成する。

    Args:
        images: 画像ファイルパスのリスト。空の場合は黒背景。
        voice_path: ナレーション音声ファイルパス。
        out_path: 出力MP4。
        size: (width, height)。デフォルト1280x720。
        max_duration: 秒指定時、音声と動画をこの長さでトリミング。
        fps: 出力フレームレート。
    """
    from moviepy.editor import AudioFileClip, ImageClip, ColorClip, concatenate_videoclips

    audio_clip = AudioFileClip(voice_path)
    # 長さ制限（Shorts/TikTok用）
    if max_duration and audio_clip.duration > max_duration:
        audio_clip = audio_clip.subclip(0, max_duration)
    duration = audio_clip.duration

    # 画像→VideoClip 化
    if images:
        img_duration = duration / len(images)
        clips: list[ImageClip] = []
        for path in images:
            clip = ImageClip(path)
            # リサイズ→クロップで指定アスペクトへ合わせる
            clip = clip.resize(height=size[1]) if clip.h < clip.w else clip.resize(width=size[0])
            clip = clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=size[0], height=size[1])
            clips.append(clip.set_duration(img_duration))
    else:
        clips = [ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)]

    video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
    video.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=fps, preset="medium", threads=4)
    return out_path


def make_video_variants(
    images: list[str],
    voice_path: str,
    base_filename: str,
    max_duration_short: int = 60,
) -> dict[str, str]:
    """横長(16:9)と縦長(9:16)の2種類を生成しファイルパスを返す。"""
    from pathlib import Path

    base = Path(base_filename).with_suffix("")
    horizontal_path = f"{base}_h.mp4"
    vertical_path = f"{base}_v.mp4"
    # 横長 1280x720 (16:9)
    make_video(images, voice_path, horizontal_path, size=(1280, 720))
    # 縦長 1080x1920 (9:16) & 60秒以内
    make_video(images, voice_path, vertical_path, size=(1080, 1920), max_duration=max_duration_short)
    return {"horizontal": horizontal_path, "vertical": vertical_path}

# ----------------------------------------------------------------------------
# 6) Google Drive アップロード
# ----------------------------------------------------------------------------

def upload_drive(file_path: str) -> str:
    """Upload video to Google Drive folder and return share URL."""
    creds = get_service_account_creds()
    drive = build("drive", "v3", credentials=creds)

    # フォルダ存在確認（shared drive も考慮して supportsAllDrives=True）
    drive.files().get(fileId=DRIVE_FOLDER_ID, fields="id", supportsAllDrives=True).execute()

    meta = {"name": Path(file_path).name, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    file = drive.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    # 公開リンク作成
    drive.permissions().create(fileId=file["id"], body={"role": "reader", "type": "anyone"}).execute()
    return f"https://drive.google.com/file/d/{file['id']}/view"

# ----------------------------------------------------------------------------
# 7) YouTube アップロード
# ----------------------------------------------------------------------------

def upload_youtube(
    file_path: str,
    title: str,
    caption: str | None = None,
    publish_at: str | None = None,
    srt_path: str | None = None,
    draft: bool = False,
    shorts: bool = False,
) -> str:
    yt_creds = get_youtube_creds()
    yt = build("youtube", "v3", credentials=yt_creds)
    # Shorts 判定用にハッシュタグ追加
    if shorts and "#shorts" not in title.lower():
        title += " #shorts"

    body = {
        "snippet": {
            "title": title,
            "description": caption or title,
            "categoryId": "25",
        },
        "status": {},
    }
    if shorts:
        body["snippet"].setdefault("tags", []).append("shorts")
    if draft:
        body["status"].update({"privacyStatus": "private"})
    elif publish_at:
        # 予約公開
        body["status"].update({"privacyStatus": "private", "publishAt": publish_at})
    else:
        body["status"].update({"privacyStatus": "public"})
    media = MediaFileUpload(file_path, resumable=True, mimetype="video/mp4")
    resp = yt.videos().insert(part="snippet,status", body=body, media_body=media).execute()

    # SRT 字幕があればアップロード
    if srt_path and Path(srt_path).exists():
        from googleapiclient.http import MediaFileUpload as _MFU
        media = _MFU(srt_path, mimetype="application/x-subrip", resumable=False)
        try:
            yt.captions().insert(
                part="snippet",
                body={
                    "snippet": {
                        "language": "ja",
                        "name": "Japanese",
                        "videoId": resp["id"],
                        "isDraft": False,
                    }
                },
                media_body=media,
            ).execute()
        except googleapiclient.errors.HttpError as e:
            # 409: 同名トラックが既に存在する場合は上書き
            if e.resp.status == 409:
                logging.info("Caption already exists – updating instead of inserting")
                # 既存トラック取得
                existing = yt.captions().list(part="id,snippet", videoId=resp["id"]).execute()
                ja_id = None
                for item in existing.get("items", []):
                    sn = item.get("snippet", {})
                    if sn.get("language") == "ja" and sn.get("name") == "Japanese":
                        ja_id = item["id"]
                        break
                if ja_id:
                    yt.captions().update(
                        part="snippet",
                        body={
                            "id": ja_id,
                            "snippet": {
                                "language": "ja",
                                "name": "Japanese",
                                "videoId": resp["id"],
                                "isDraft": False,
                            },
                        },
                        media_body=media,
                    ).execute()
                else:
                    # 同名は無いが409? → nameを付け替えて再挿入
                    yt.captions().insert(
                        part="snippet",
                        body={
                            "snippet": {
                                "language": "ja",
                                "name": f"Japanese_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                                "videoId": resp["id"],
                                "isDraft": False,
                            }
                        },
                        media_body=media,
                    ).execute()
            else:
                logging.exception("upload_youtube: SRT upload failed")

        return f"https://youtu.be/{resp['id']}"


# ----------------------------------------------------------------------------
# 7-2) Instagram / TikTok / Twitter アップロード (スタブ)
# ----------------------------------------------------------------------------

def _missing_cred_msg(service: str, envs: list[str]):
    return f"{service} アクセスに必要な環境変数が不足しています: {', '.join(envs)}"


def upload_instagram(file_path: str, caption: str) -> str:
    """Instagram Reels へ動画を投稿 (スタブ)。Env が無ければ空文字を返す。"""
    required = ["IG_ACCESS_TOKEN", "IG_USER_ID"]
    if any(not os.environ.get(v) for v in required):
        logging.warning(_missing_cred_msg("Instagram", required))
        return ""
    # 本実装では Graph API の /media, /media_publish を使用
    # TODO: implement
    logging.info("upload_instagram: stub called, returning placeholder URL")
    return "https://instagram.com/reel/PLACEHOLDER"


def upload_tiktok(file_path: str, caption: str) -> str:
    """TikTok へ動画を投稿 (スタブ)。"""
    required = ["TIKTOK_ACCESS_TOKEN", "TIKTOK_CLIENT_KEY"]
    if any(not os.environ.get(v) for v in required):
        logging.warning(_missing_cred_msg("TikTok", required))
        return ""
    # TODO: implement TikTok Upload API
    logging.info("upload_tiktok: stub called, returning placeholder URL")
    return "https://tiktok.com/@user/video/PLACEHOLDER"


def upload_twitter(file_path: str, caption: str) -> str:
    """Twitter(X) へ動画を投稿 (スタブ)。"""
    required = ["TWITTER_BEARER_TOKEN", "TWITTER_API_KEY", "TWITTER_API_SECRET"]
    if any(not os.environ.get(v) for v in required):
        logging.warning(_missing_cred_msg("Twitter", required))
        return ""
    # TODO: implement Twitter v2 media upload + tweet
    logging.info("upload_twitter: stub called, returning placeholder URL")
    return "https://twitter.com/user/status/PLACEHOLDER"

# ----------------------------------------------------------------------------
# 8) スプレッドシート更新
# ----------------------------------------------------------------------------

def update_sheet(
    row: int,
    drive_url: str,
    yt_url: str,
    yt_short_url: str | None,
    ig_url: str | None,
    tiktok_url: str | None,
    tw_url: str | None,
    script: str,
    article: str,
):
    """指定行の各セルを更新し、生成フラグを 2 にセットする。"""
    sheet = get_sheet()
    if drive_url:
        sheet.update(f"B{row}", drive_url)
    if yt_url:
        sheet.update(f"C{row}", yt_url)
    if yt_short_url:
        sheet.update(f"I{row}", yt_short_url)
    if ig_url:
        sheet.update(f"J{row}", ig_url)
    if tiktok_url:
        sheet.update(f"K{row}", tiktok_url)
    if tw_url:
        sheet.update(f"L{row}", tw_url)
    if article:
        sheet.update(f"G{row}", article[:5000])
    if script:
        sheet.update(f"H{row}", script[:5000])
    # フラグを 2 へ
    sheet.update(f"E{row}", "2")

# ----------------------------------------------------------------------------
# メイン処理関数
# ----------------------------------------------------------------------------

def process_row(row: int, url: str, publish_at: str | None = None, publish_date_raw: str | None = None, draft: bool = False):
    sheet = get_sheet()
    article_title, article = fetch_article(url)

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
        script = clean_narration(create_narration(article_title + "\n" + article))
    except Exception as e:
        sheet.update(f"D{row}", f"原稿生成失敗: {e}")
        raise

    # 判定: 原稿が無い、または画像が無いのに生成無効フラグが立っていない場合のみスキップ
    missing_images_unexpected = (not images) and (not DISABLE_IMAGE_GEN)
    if missing_images_unexpected or not script.strip():
        reason = []
        if missing_images_unexpected:
            reason.append("画像生成失敗")
        if not script.strip():
            reason.append("原稿なし")
        sheet.update(f"D{row}", "/".join(reason) or "スキップ")
        logging.info("process_row: row %d skipped (%s)", row, reason)
        return {"row": row, "skipped": True, "reason": reason}

    # 音声合成 (単語タイムポイント取得付き)
    voice, tps = synthesize_speech(script, with_timepoints=True)

    try:
        from moviepy.editor import AudioFileClip as _AFC
        duration = _AFC(voice).duration
    except Exception:
        logging.exception("Failed to open voice file for duration, fallback 0")
        duration = 0.0
    # --------------------
    # 動画ファイル名決定
    # --------------------
    # 日付取得（引数優先）
    raw_date = (publish_date_raw or "").strip() or (sheet.acell(f"F{row}").value or "").strip()
    from datetime import datetime
    date_yyyymmdd = ""
    if raw_date:
        try:
            dt = datetime.fromisoformat(raw_date.replace("/", "-"))
            date_yyyymmdd = dt.strftime("%Y%m%d")
        except Exception:
            # 数字抽出 fallback
            import re
            m = re.findall(r"\d", raw_date)
            if len(m) >= 8:
                date_yyyymmdd = "".join(m)[:8]
    if not date_yyyymmdd:
        date_yyyymmdd = datetime.now().strftime("%Y%m%d")
    title_raw = article_title or script.split("\n")[0]
    title = title_raw[:50] if title_raw else "ニュース"
    base_path = f"/tmp/{_sanitize_filename(f'{date_yyyymmdd}_{title}')}.mp4"
    videos = make_video_variants(images, voice, base_path)
    video_path = videos["horizontal"]
    short_path = videos["vertical"]

    # SRT 生成（横長動画と同じファイル名）
    srt_path = str(Path(video_path).with_suffix(".srt"))
    generate_srt_file(script, duration, out_path=srt_path, timepoints=tps)

    # アップロード
    drive_url = upload_drive(video_path)
    caption_text = article
    yt_url = upload_youtube(
        video_path,
        title,
        caption=caption_text,
        publish_at=publish_at,
        srt_path=srt_path,
        draft=draft,
    )
    # Shorts アップロード
    yt_short_url = upload_youtube(
        short_path,
        title,
        caption=caption_text,
        publish_at=publish_at,
        srt_path=None,
        draft=draft,
        shorts=True,
    )
    # Instagram / TikTok / Twitter
    ig_url = upload_instagram(short_path, caption_text)
    tiktok_url = upload_tiktok(short_path, caption_text)
    tw_url = upload_twitter(short_path, caption_text)

    update_sheet(row, drive_url, yt_url, yt_short_url, ig_url, tiktok_url, tw_url, script, article)
    return {
        "row": row,
        "drive": drive_url,
        "yt": yt_url,
        "yt_shorts": yt_short_url,
        "instagram": ig_url,
        "tiktok": tiktok_url,
        "twitter": tw_url,
        "srt": srt_path,
    }

# ----------------------------------------------------------------------------
# Flask ルーティング
# ----------------------------------------------------------------------------

@app.route("/process", methods=["POST"])
def process():
    logging.info("/process: start processing spreadsheet rows")
    results = []
    sheet = get_sheet()
    rows = sheet.get_all_values()[1:]  # ヘッダー除外
    for idx, r in enumerate(rows, start=2):
        r = (r + [""] * 8)[:8]
        url, exec_flag, publish_date = r[0], r[4].strip(), r[5]
        publish_at = None
        if publish_date:
            from datetime import datetime, timezone, timedelta, time as dtime
            def _parse_date(s: str):
                s = s.strip()
                for sep in ("-", "/", "."):
                    if sep in s:
                        parts = s.split(sep)
                        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                            y, m, d = map(int, parts[:3])
                            return datetime(y, m, d)
                if len(s) >= 8 and s.isdigit():
                    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
                return None
            dt_base = _parse_date(publish_date)
            if dt_base:
                # デフォルト公開時刻 10:00 JST
                dt = dt_base.replace(hour=10)  # naive
                dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
                now_jst = datetime.now(tz=timezone(timedelta(hours=9)))
                if dt <= now_jst:
                    dt = now_jst + timedelta(minutes=30)  # 30分後に調整
                publish_at = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            else:
                logging.warning("Invalid publish_date format: %s", publish_date)
        if url and exec_flag == "1":
            try:
                draft_flag = not publish_date.strip()
                res = process_row(idx, url, publish_at, publish_date, draft=draft_flag)
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
