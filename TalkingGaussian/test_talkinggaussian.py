"""
TalkingGaussian 视频生成脚本
"""

import sys
import os
import argparse
import subprocess
import shutil
import numpy as np
from datetime import datetime

def ffprobe_duration(path: str) -> float:
    """返回媒体时长（秒）"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    return float(r.stdout.strip())

def trim_video_to_duration(input_mp4: str, out_mp4: str, duration_s: float):
    """把视频裁到 duration_s（秒）。为避免关键帧问题，直接重编码更稳。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        out_mp4
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {r.stderr}")

def mux_audio(video_mp4: str, audio_wav: str, out_mp4: str):
    """给视频加音轨，输出最终 mp4"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_mp4,
        "-i", audio_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_mp4
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr}")

def ensure_audio_sample_rate(audio_file: str, target_sr: int = 16000) -> str:
    """
    检查音频采样率，如果不是 target_sr 则重采样并保存为临时文件。
    返回最终使用的音频文件路径。
    """
    try:
        # 使用 ffprobe 检查采样率
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_file
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"警告: 无法检查采样率: {r.stderr}")
            return audio_file
            
        current_sr = int(r.stdout.strip())
        
        if current_sr == target_sr:
            return audio_file
            
        print(f"检测到音频采样率为 {current_sr}Hz，正在重采样为 {target_sr}Hz...")
        
        # 生成临时文件名
        dir_name = os.path.dirname(audio_file)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        new_audio_file = os.path.join(dir_name, f"{base_name}_{target_sr}.wav")
        
        # 使用 ffmpeg 重采样
        cmd_resample = [
            "ffmpeg", "-y",
            "-i", audio_file,
            "-ar", str(target_sr),
            "-ac", "1", # 强制单声道，DeepSpeech通常需要
            new_audio_file
        ]
        r = subprocess.run(cmd_resample, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"警告: 重采样失败: {r.stderr}")
            return audio_file
            
        return new_audio_file
        
    except Exception as e:
        print(f"警告: 音频预处理出错: {e}")
        return audio_file

def extract_audio_features(audio_file, output_dir, audio_basename, extractor, ts: str):
    """提取音频特征"""
    
    # 预处理：确保音频是 16k 采样率
    processed_audio_file = ensure_audio_sample_rate(audio_file, 16000)
    
    if extractor == 'deepspeech':
        # DeepSpeech
        feature_file = os.path.join(output_dir, f"{audio_basename}_{ts}.npy")
        extract_script = 'data_utils/deepspeech_features/extract_ds_features.py'
        extract_args = ['--input', processed_audio_file, '--output', feature_file]
    else:  # hubert
        feature_file = os.path.join(output_dir, f"{audio_basename}_{ts}_hu.npy")
        extract_script = 'data_utils/hubert.py'
        extract_args = ['--wav', processed_audio_file]
        
    # 提取特征
    print(f"正在提取{extractor}音频特征...")
    cmd = ['python', extract_script] + extract_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {extractor}特征提取失败: {result.stderr}")
        if extractor == 'deepspeech':
            print("提示: DeepSpeech特征提取遇到问题，将尝试使用HuBERT替代")
        return None

    # HuBERT 脚本固定输出到 <wav>_hu.npy，这里把它搬运/重命名到带时间戳的路径
    if extractor != 'deepspeech':
        hubert_default = os.path.splitext(audio_file)[0] + '_hu.npy'
        if os.path.exists(hubert_default):
            shutil.copy2(hubert_default, feature_file)
        
    # 检查特征文件是否生成成功
    if not os.path.exists(feature_file):
        print(f"错误: 特征文件 {feature_file} 未生成")
        return None
        
    return feature_file

def main():
    parser = argparse.ArgumentParser(description='TalkingGaussian 视频生成工具')
    parser.add_argument('--audio_file', 
                        default='test_result/cloned_voice_0.wav',
                        help='输入的.wav音频文件路径')
    parser.add_argument('--dataset_path',
                        default='data/May',
                        help='数据集路径，包含头部姿态等必要信息')
    parser.add_argument('--model_path',
                        default='output/talking_May',
                        help='训练好的模型路径')
    parser.add_argument('--output_dir',
                        default='test_result',
                        help='输出目录 ')
    parser.add_argument('--audio_extractor',
                        choices=['deepspeech', 'hubert'],
                        default='deepspeech',
                        help='使用的音频特征提取器')
    parser.add_argument('--sh_degree',
                        type=int,
                        choices=[0, 1, 2, 3],
                        default=2,
                        help='渲染质量等级：0(极速), 1(标准), 2(高保真), 3(最高保真)')

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件 {args.audio_file} 不存在")
        return

    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在")
        return

    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 {args.model_path} 不存在")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 提取音频文件名作为基础名称
    audio_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
    
    # 提取音频特征
    feature_file = extract_audio_features(args.audio_file, args.output_dir, audio_basename, args.audio_extractor, ts)
    
    # 如果首选特征提取失败，尝试备用方法
    if feature_file is None:
        alternative_extractor = 'hubert' if args.audio_extractor == 'deepspeech' else 'deepspeech'
        print(f"尝试使用{alternative_extractor}作为替代特征提取器...")
        feature_file = extract_audio_features(args.audio_file, args.output_dir, audio_basename, alternative_extractor, ts)
    
    if feature_file is None:
        print("错误: 无法提取音频特征，两种方法均失败")
        return

    print(f"音频特征准备完成: {feature_file}")

    # --- 调试与强制修正逻辑开始 ---
    try:
        feats = np.load(feature_file)
        print(f"特征文件形状: {feats.shape}")
        num_frames = feats.shape[0]
        predicted_dur = num_frames / 25.0
        print(f"特征帧数: {num_frames}, 预计视频时长: {predicted_dur:.2f} 秒")
        
        # 获取音频真实时长
        real_audio_dur = ffprobe_duration(args.audio_file)
        print(f"音频真实时长: {real_audio_dur:.2f} 秒")
        
        # 检查误差 (允许 0.5秒 或 12帧 的误差)
        if predicted_dur > real_audio_dur + 0.5:
            print(f"警告: 特征时长 ({predicted_dur:.2f}s) 显著长于音频 ({real_audio_dur:.2f}s)")
            expected_frames = int(real_audio_dur * 25) + 1 # 多留1帧防截断
            print(f"正在强制截断特征至 {expected_frames} 帧...")
            
            # 截断并覆盖保存
            feats_trimmed = feats[:expected_frames]
            np.save(feature_file, feats_trimmed)
            print(f"特征已修正，新形状: {feats_trimmed.shape}")
            
    except Exception as e:
        print(f"警告: 特征检查/修正失败: {e}")
    # --- 调试与强制修正逻辑结束 ---

    # 生成视频
    print("正在生成人脸说话视频...")
    use_train = True  # 将参数提取为变量
    cmd = [
        'python', 'synthesize_fuse.py',
        '-S', args.dataset_path,
        '-M', args.model_path,
        '--audio', feature_file,
        '--sh_degree', str(args.sh_degree),
    ]
    if use_train:
        cmd.append('--use_train')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ...existing code...

    # 查找生成的视频文件
    # 根据 use_train 决定子目录是 'train' 还是 'test'
    sub_dir = "train" if use_train else "test"
    render_output_dir = os.path.join(args.model_path, sub_dir, "ours_None", "renders")
    generated_video = os.path.join(render_output_dir, "out.mp4")
    
    if not os.path.exists(generated_video):
        print("警告: 未能找到生成的视频文件")
        # 尝试查找其他可能的位置
        alternative_paths = [
            os.path.join(args.model_path, "train", "ours_None", "renders", "out.mp4"),
        ]
        
        found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                generated_video = alt_path
                found = True
                break
                
        if not found:
            print("错误: 在预期位置未找到生成的视频文件")
            return
    
    # 将视频复制到输出目录并重命名
    output_video = os.path.join(args.output_dir, f"{audio_basename}_talking_head.mp4")
    shutil.copy2(generated_video, output_video)
    
    print(f"生成无声视频: {output_video}")
    
    # 检查生成视频的实际时长
    try:
        actual_dur = ffprobe_duration(output_video)
        print(f"生成视频实际时长: {actual_dur:.3f}s")
    except Exception as e:
        print(f"无法获取视频时长: {e}")

    # 1) 读音频时长
    audio_dur = ffprobe_duration(args.audio_file)
    print(f"输入音频时长: {audio_dur:.3f}s")

    # 2) 裁剪视频到音频时长
    trimmed_video = os.path.join(args.output_dir, f"{audio_basename}_trimmed.mp4")
    trim_video_to_duration(output_video, trimmed_video, audio_dur)
    print(f"裁剪后视频: {trimmed_video}")

    # 3) 合成音轨
    final_video = os.path.join(args.output_dir, f"{audio_basename}_final.mp4")
    mux_audio(trimmed_video, args.audio_file, final_video)
    print(f"最终有声视频: {final_video}")

        
    print(f"人脸说话视频生成完成: {output_video}")


if __name__ == '__main__':
    main()