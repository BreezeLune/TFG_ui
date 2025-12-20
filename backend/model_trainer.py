import subprocess
import os
import time

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    video_path = data['ref_video']
    print(f"输入视频：{video_path}")

    print("[backend.model_trainer] 模型训练中...")

    if data['model_choice'] == "TalkingGaussian":
        try:
            # 获取参数
            video_path = data['ref_video']
            gpu_choice = data.get('gpu_choice', 'GPU0')
            epochs = data.get('epoch', '1000')
            audio_extractor = data.get('audio_extractor', 'deepspeech')
            
            # 从视频路径推断数据目录
            # 假设视频路径格式: TalkingGaussian/data/<ID>/<ID>.mp4
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 如果视频路径不包含完整路径，尝试构建
            if not os.path.isabs(video_path) and not video_path.startswith('TalkingGaussian/'):
                dataset_path = os.path.join("TalkingGaussian", "data", video_name)
            else:
                dataset_path = video_dir
            
            # 检查数据是否已预处理（检查 transforms_train.json）
            transforms_file = os.path.join(dataset_path, "transforms_train.json")
            if not os.path.exists(transforms_file):
                print("[backend.model_trainer] 数据未预处理，开始预处理...")
                # 执行数据预处理
                preprocess_cmd = [
                    'python', 'TalkingGaussian/data_utils/process.py',
                    video_path,
                    '--asr', audio_extractor
                ]
                
                preprocess_result = subprocess.run(
                    preprocess_cmd,
                    capture_output=True,
                    text=True,
                    cwd='.'
                )
                
                if preprocess_result.returncode != 0:
                    print(f"[backend.model_trainer] 预处理失败: {preprocess_result.stderr}")
                    return video_path
                
                print("[backend.model_trainer] 数据预处理完成")
            
            # 设置输出目录
            workspace = os.path.join("TalkingGaussian", "output", video_name)
            os.makedirs(workspace, exist_ok=True)
            
            # 解析GPU ID
            gpu_id = gpu_choice.replace('GPU', '') if 'GPU' in gpu_choice else '0'
            
            # 构建训练命令 - 使用训练脚本
            cmd = [
                'bash', 'TalkingGaussian/scripts/train_xx.sh',
                dataset_path,      # 数据目录
                workspace,        # 输出目录
                gpu_id            # GPU ID
            ]
            
            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            
            # 设置GPU环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id
            
            # 执行训练命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd='.'
            )
            
            print("[backend.model_trainer] 训练输出:", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] 错误输出:", result.stderr)
            
            if result.returncode == 0:
                print(f"[backend.model_trainer] 训练完成，模型保存在: {workspace}")
                # 返回相对路径格式，便于后续使用
                # workspace 是 "TalkingGaussian/output/May"，返回 "output/May"
                relative_workspace = os.path.relpath(workspace, "TalkingGaussian")
                if relative_workspace.startswith('..'):
                    # 如果相对路径计算失败，使用简单方法
                    relative_workspace = workspace.replace('TalkingGaussian/', '').lstrip('/')
                print(f"[backend.model_trainer] 返回相对路径: {relative_workspace}")
                return relative_workspace
            else:
                print(f"[backend.model_trainer] 训练失败，退出码: {result.returncode}")
                return video_path
                
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            return video_path
    
    elif data['model_choice'] == "SyncTalk":
        try:
            # 构建命令
            cmd = [
                "./SyncTalk/run_synctalk.sh", "train",
                "--video_path", data['ref_video'],
                "--gpu", data['gpu_choice'],
                "--epochs", data['epoch']
            ]
            
            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            # 执行训练命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("[backend.model_trainer] 训练输出:", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] 错误输出:", result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] 训练失败，退出码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return video_path
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return video_path

    print("[backend.model_trainer] 训练完成")
    return video_path
