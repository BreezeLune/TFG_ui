dataset=$1
workspace=$2
gpu_id=$3
audio_extractor='deepspeech' # deepspeech, esperanto, hubert

export CUDA_VISIBLE_DEVICES=$gpu_id

# Get script directory and change to TalkingGaussian directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TALKINGGAUSSIAN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TALKINGGAUSSIAN_DIR" || exit 1

# Convert paths to relative paths (relative to TalkingGaussian directory)
# If dataset or workspace paths contain "TalkingGaussian/", remove that prefix
dataset_rel="${dataset#TalkingGaussian/}"
workspace_rel="${workspace#TalkingGaussian/}"

# Use conda run to switch to talking_gaussian environment for all Python commands
conda run -n talking_gaussian --no-capture-output python train_mouth.py -s "$dataset_rel" -m "$workspace_rel" --audio_extractor "$audio_extractor"
conda run -n talking_gaussian --no-capture-output python train_face.py -s "$dataset_rel" -m "$workspace_rel" --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor "$audio_extractor"
conda run -n talking_gaussian --no-capture-output python train_fuse.py -s "$dataset_rel" -m "$workspace_rel" --opacity_lr 0.001 --audio_extractor "$audio_extractor"

# # Parallel. Ensure that you have aleast 2 GPUs, and over N x 64GB memory for about N x 5k frames (IMPORTANT! Otherwise the computer will crash).
# CUDA_VISIBLE_DEVICES=$gpu_id conda run -n talking_gaussian --no-capture-output python train_mouth.py -s "$dataset_rel" -m "$workspace_rel" --audio_extractor "$audio_extractor" & 
# CUDA_VISIBLE_DEVICES=$((gpu_id+1)) conda run -n talking_gaussian --no-capture-output python train_face.py -s "$dataset_rel" -m "$workspace_rel" --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor "$audio_extractor"
# CUDA_VISIBLE_DEVICES=$gpu_id conda run -n talking_gaussian --no-capture-output python train_fuse.py -s "$dataset_rel" -m "$workspace_rel" --opacity_lr 0.001 --audio_extractor "$audio_extractor"

conda run -n talking_gaussian --no-capture-output python synthesize_fuse.py -s "$dataset_rel" -m "$workspace_rel" --eval --audio_extractor "$audio_extractor"
conda run -n talking_gaussian --no-capture-output python metrics.py "$workspace_rel/test/ours_None/renders/out.mp4" "$workspace_rel/test/ours_None/gt/out.mp4"