#!/usr/bin/env bash
set -e

# Usage:
#   ./run_cosyvoice.sh --model_dir <dir> --prompt_wav <wav> --prompt_text <text> --tts_text <text> --language <zh|en> --speed <speed> --output_file <file>

# Parse arguments
MODEL_DIR=""
PROMPT_WAV=""
PROMPT_TEXT=""
TTS_TEXT=""
LANGUAGE="en"
SPEED="1.0"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dir) MODEL_DIR="$2"; shift 2;;
    --prompt_wav) PROMPT_WAV="$2"; shift 2;;
    --prompt_text) PROMPT_TEXT="$2"; shift 2;;
    --tts_text) TTS_TEXT="$2"; shift 2;;
    --language) LANGUAGE="$2"; shift 2;;
    --speed) SPEED="$2"; shift 2;;
    --output_file) OUTPUT_FILE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [ -z "$MODEL_DIR" ] || [ -z "$PROMPT_WAV" ] || [ -z "$TTS_TEXT" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Missing required arguments: --model_dir, --prompt_wav, --tts_text, --output_file"
  exit 1
fi

# Build command arguments
CMD_ARGS=(
  "--model_dir" "$MODEL_DIR"
  "--prompt_wav" "$PROMPT_WAV"
  "--tts_text" "$TTS_TEXT"
  "--output_file" "$OUTPUT_FILE"
)

if [ -n "$PROMPT_TEXT" ]; then
  CMD_ARGS+=("--prompt_text" "$PROMPT_TEXT")
fi

if [ -n "$LANGUAGE" ]; then
  CMD_ARGS+=("--language" "$LANGUAGE")
fi

if [ -n "$SPEED" ]; then
  CMD_ARGS+=("--speed" "$SPEED")
fi

# Use conda run to switch to cosyvoice environment
# 注意：run_cosyvoice.sh 是在项目根目录下通过 `bash ./CosyVoice/run_cosyvoice.sh` 调用，
# 所以这里需要显式加上 CosyVoice 子目录，否则会在 `/root/TFG_ui` 下找不到脚本。
conda run -n cosyvoice --no-capture-output \
    python ./CosyVoice/test_cosyvoice.py "${CMD_ARGS[@]}"

echo "OK: CosyVoice synthesis completed"

