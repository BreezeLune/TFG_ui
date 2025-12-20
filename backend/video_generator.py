import os
import time
import subprocess
import shutil
from pathlib import Path

def extract_relative_path(path_str, required_folder=None):
    """
    从各种路径格式中提取相对路径（相对于 TalkingGaussian 目录）
    
    Args:
        path_str: 输入路径，可能是 "output/May"、"TalkingGaussian/output/May"、"/app/TalkingGaussian/output/May" 等
        required_folder: 必须包含的文件夹名（如 "output" 或 "data"），如果找不到则返回 None
    
    Returns:
        提取的相对路径，如 "output/May" 或 "data/May"（统一使用正斜杠）
    """
    if not path_str:
        return None
    
    # 使用 Path 处理，自动处理 Windows/Linux 路径差异
    parts = Path(path_str).parts
    
    # 如果指定了必须包含的文件夹，检查是否存在
    if required_folder:
        if required_folder not in parts:
            return None
        # 从找到的文件夹位置开始截取
        folder_idx = parts.index(required_folder)
        # 使用 as_posix() 确保返回正斜杠路径（跨平台兼容）
        return Path(*parts[folder_idx:]).as_posix()
    
    # 如果没有指定，尝试自动识别 output 或 data
    for folder in ('output', 'data'):
        if folder in parts:
            folder_idx = parts.index(folder)
            # 使用 as_posix() 确保返回正斜杠路径（跨平台兼容）
            return Path(*parts[folder_idx:]).as_posix()
    
    # 如果都找不到，返回原路径（可能是已经是相对路径）
    # 使用 as_posix() 确保返回正斜杠路径（跨平台兼容）
    return Path(*parts).as_posix()

def validate_model_path(model_path):
    """
    验证模型路径是否存在且包含必要的检查点文件
    """
    if not model_path:
        return False, None, "模型路径为空"
    
    # 提取相对路径（必须包含 output）
    normalized_path = extract_relative_path(model_path, required_folder='output')
    if not normalized_path:
        return False, None, f"模型路径格式错误，必须包含 'output' 文件夹: {model_path}"
    
    # 构建完整路径
    full_path = os.path.join("TalkingGaussian", normalized_path)
    
    # 检查目录是否存在
    if not os.path.exists(full_path):
        return False, normalized_path, f"模型目录不存在: {full_path}"
    
    # 检查检查点文件（使用 any 简化逻辑）
    essentials = [
        "chkpnt_fuse_latest.pth",
        "chkpnt_face_latest.pth", 
        "chkpnt_mouth_latest.pth"
    ]
    if not any(os.path.exists(os.path.join(full_path, f)) for f in essentials):
        return False, normalized_path, f"模型检查点文件不存在于: {full_path}"
    
    return True, normalized_path, None

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if data['model_name'] == "TalkingGaussian":
        try:
            # 获取参数
            audio_path = data['ref_audio']
            model_path_raw = data['model_param']  # 可能是各种格式
            dataset_path_raw = data.get('dataset_path', 'TalkingGaussian/data/May')  # 默认数据目录
            audio_extractor = data.get('audio_extractor', 'deepspeech')  # deepspeech 或 hubert
            gpu_choice = data.get('gpu_choice', 'GPU0')
            
            # 获取推理参数（方案二：渲染细节等级）
            inference_params = data.get('inference_params', {})
            sh_degree = inference_params.get('sh_degree', 2)  # 默认值 2（标准模式）
            # 验证 sh_degree 范围
            if sh_degree not in [0, 1, 2, 3]:
                print(f"[backend.video_generator] 警告：sh_degree={sh_degree} 不在有效范围 [0,1,2,3] 内，使用默认值 2")
                sh_degree = 2
            
            # 验证模型路径（自动提取相对路径）
            is_valid, model_path, error_msg = validate_model_path(model_path_raw)
            if not is_valid:
                print(f"[backend.video_generator] 模型路径验证失败: {error_msg}")
                return os.path.join("static", "videos", "out.mp4")
            
            # 处理数据目录路径（自动提取相对路径）
            dataset_path = extract_relative_path(dataset_path_raw, required_folder='data')
            if not dataset_path:
                dataset_path = 'data/May'  # 默认值
            
            # 生成输出视频文件名
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            video_filename = f"talkinggaussian_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # 构建命令 - 使用 run_talkinggaussian.sh 封装脚本
            # 注意：model_path 和 dataset_path 已经是统一后的相对路径格式
            cmd = [
                'bash', 'TalkingGaussian/run_talkinggaussian.sh', 'infer',
                '--wav', audio_path,
                '--out', destination_path,
                '--extractor', audio_extractor,
                '--dataset', dataset_path,
                '--model', model_path,
                '--sh_degree', str(sh_degree)  # 渲染细节等级（方案二）
            ]
            
            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")
            
            # 设置GPU环境变量（如果需要）
            env = os.environ.copy()
            if gpu_choice and gpu_choice != "CPU":
                gpu_id = gpu_choice.replace('GPU', '') if 'GPU' in gpu_choice else '0'
                env['CUDA_VISIBLE_DEVICES'] = gpu_id
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd='.'  # 在项目根目录执行
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)
            
            # 检查输出文件是否存在
            if os.path.exists(destination_path):
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path
            else:
                # 尝试查找 test_result 目录中的文件
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                test_result_dir = "TalkingGaussian/test_result"
                if os.path.exists(test_result_dir):
                    # 查找最新的 final.mp4 文件
                    final_files = [f for f in os.listdir(test_result_dir) 
                                 if f.startswith(audio_basename) and f.endswith('_final.mp4')]
                    if final_files:
                        latest_file = max(final_files, key=lambda f: os.path.getctime(os.path.join(test_result_dir, f)))
                        source_path = os.path.join(test_result_dir, latest_file)
                        shutil.copy(source_path, destination_path)
                        print(f"[backend.video_generator] 找到生成的视频文件: {destination_path}")
                        return destination_path
                
                print(f"[backend.video_generator] 视频文件不存在: {destination_path}")
                return os.path.join("static", "videos", "out.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            import traceback
            traceback.print_exc()
            return os.path.join("static", "videos", "out.mp4")
    
    elif data['model_name'] == "SyncTalk":
        try:
            
            # 构建命令
            cmd = [
                './SyncTalk/run_synctalk.sh', 'infer',
                '--model_dir', data['model_param'],
                '--audio_path', data['ref_audio'],
                '--gpu', data['gpu_choice']
            ]

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # check=True
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)
            
            # 文件原路径与目的路径 
            model_dir_name = os.path.basename(data['model_param'])
            source_path = os.path.join("SyncTalk", "model", model_dir_name, "results", "test_audio.mp4")
            audio_name = os.path.splitext(os.path.basename(data['ref_audio']))[0]
            video_filename = f"{model_dir_name}_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)
            # 检查文件是否存在
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path
            else:
                print(f"[backend.video_generator] 视频文件不存在: {source_path}")
                # 尝试查找任何新生成的mp4文件
                results_dir = os.path.join("SyncTalk", "model", model_dir_name, "results")
                if os.path.exists(results_dir):
                    mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                    if mp4_files:
                        latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                        source_path = os.path.join(results_dir, latest_file)
                        shutil.copy(source_path, destination_path)
                        print(f"[backend.video_generator] 找到最新视频文件: {destination_path}")
                        return destination_path
                
                return os.path.join("static", "videos", "out.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")
    
    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path
