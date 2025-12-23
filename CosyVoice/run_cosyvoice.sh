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

# 固定工作目录到 CosyVoice 目录，输出统一放在 CosyVoice/test_result 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUTPUT_DIR="$SCRIPT_DIR/test_result"
mkdir -p "$OUTPUT_DIR"
# 确保输出文件落在 CosyVoice/test_result 内
OUTPUT_FILE="$OUTPUT_DIR/$(basename "$OUTPUT_FILE")"

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
conda run -n cosyvoice --no-capture-output \
    python ./test_cosyvoice.py "${CMD_ARGS[@]}"

echo "OK: CosyVoice synthesis completed"

