import subprocess
import os
import time
import shutil


def _resp(status, model_path=None, message=""):
    return {
        "status": status,
        "model_path": model_path,
        "message": message
    }


def train_model(data):
    """
    模型训练逻辑，统一返回:
    {
        'status': 'success' | 'error',
        'model_path': str | None,
        'message': str
    }
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

            # 提取视频名称
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # 方案B：如果视频在 static/uploads/videos/，自动复制到 TalkingGaussian/data/
            original_video_path = video_path
            if video_path.startswith('static/uploads/videos/') or video_path.startswith('./static/uploads/videos/'):
                print(f"[backend.model_trainer] 检测到上传目录的视频，自动复制到训练目录...")
                # 构建目标目录
                dataset_path = os.path.join("TalkingGaussian", "data", video_name)
                os.makedirs(dataset_path, exist_ok=True)

                # 目标视频路径
                target_video_path = os.path.join(dataset_path, os.path.basename(video_path))

                # 如果目标文件不存在或源文件更新，则复制
                if not os.path.exists(target_video_path) or \
                        (os.path.exists(video_path) and os.path.getmtime(video_path) > os.path.getmtime(
                            target_video_path)):
                    shutil.copy2(video_path, target_video_path)
                    print(f"[backend.model_trainer] 已复制视频: {video_path} -> {target_video_path}")
                else:
                    print(f"[backend.model_trainer] 目标视频已存在且较新，跳过复制")

                # 更新 video_path 为训练目录中的路径
                video_path = target_video_path
            else:
                # 原有逻辑：从视频路径推断数据目录
                video_dir = os.path.dirname(video_path)

            # 如果视频路径不包含完整路径，尝试构建
            if not os.path.isabs(video_path) and not video_path.startswith('TalkingGaussian/'):
                dataset_path = os.path.join("TalkingGaussian", "data", video_name)
            else:
                dataset_path = video_dir

            # 检查数据是否已预处理（检查 transforms_train.json）
            transforms_file = os.path.join(dataset_path, "transforms_train.json")
            if not os.path.exists(transforms_file):
                print("[backend.model_trainer] 数据未预处理，开始预处理...")
                # 执行数据预处理 - 使用conda run切换到talking_gaussian环境
                preprocess_cmd = [
                    'conda', 'run', '-n', 'talking_gaussian', '--no-capture-output',
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
                    return _resp("error", None, f"预处理失败: {preprocess_result.stderr}")

                print("[backend.model_trainer] 数据预处理完成")

            # 设置输出目录
            workspace = os.path.join("TalkingGaussian", "output", video_name)
            os.makedirs(workspace, exist_ok=True)

            # 解析GPU ID
            gpu_id = gpu_choice.replace('GPU', '') if 'GPU' in gpu_choice else '0'

            # 构建训练命令 - 使用训练脚本
            cmd = [
                'bash', 'TalkingGaussian/scripts/train_xx.sh',
                dataset_path,  # 数据目录
                workspace,  # 输出目录
                gpu_id  # GPU ID
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

                # 方案B：训练完成后，自动复制提取的音频到 static/uploads/audios/
                extracted_audio_path = os.path.join(dataset_path, "aud.wav")
                if os.path.exists(extracted_audio_path):
                    uploads_audio_dir = "static/uploads/audios"
                    os.makedirs(uploads_audio_dir, exist_ok=True)
                    reference_audio_path = os.path.join(uploads_audio_dir, f"{video_name}_reference.wav")
                    shutil.copy2(extracted_audio_path, reference_audio_path)
                    print(f"[backend.model_trainer] 已复制提取的音频到: {reference_audio_path}")
                else:
                    print(f"[backend.model_trainer] 警告: 未找到提取的音频文件: {extracted_audio_path}")

                # 返回相对路径格式，便于后续使用
                # workspace 是 "TalkingGaussian/output/May"，返回 "output/May"
                relative_workspace = os.path.relpath(workspace, "TalkingGaussian")
                if relative_workspace.startswith('..'):
                    # 如果相对路径计算失败，使用简单方法
                    relative_workspace = workspace.replace('TalkingGaussian/', '').lstrip('/')
                print(f"[backend.model_trainer] 返回相对路径: {relative_workspace}")
                return _resp("success", relative_workspace, "训练完成")
            else:
                print(f"[backend.model_trainer] 训练失败，退出码: {result.returncode}")
                return _resp("error", None, f"训练失败，退出码: {result.returncode}")

        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return _resp("error", None, "找不到训练脚本")
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            return _resp("error", None, f"未知错误: {e}")

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
            return _resp("error", None, f"训练失败: {e.stderr}")
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return _resp("error", None, "找不到训练脚本")
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return _resp("error", None, f"未知错误: {e}")

    print("[backend.model_trainer] 训练完成")
    return _resp("success", None, "训练完成")
