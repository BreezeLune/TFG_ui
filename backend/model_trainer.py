import subprocess
import os
import time
import shutil

def _resp(status, model_path=None, message="", preview_video=None, reference_audio=None, reference_audio_short=None):
    return {
        "status": status,
        "model_path": model_path,
        "message": message,
        "preview_video": preview_video,
        "reference_audio": reference_audio,
        "reference_audio_short": reference_audio_short
    }


def _trim_audio(src_path, dst_path, max_seconds=29.5):
    """
    使用 ffmpeg 将音频裁剪到指定秒数。
    返回输出路径（若失败则返回 None）。
    """
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", src_path,
            "-t", f"{max_seconds}",
            "-c:a", "pcm_s16le",
            dst_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[backend.model_trainer] 裁剪音频失败: {result.stderr}")
            return None
        return dst_path
    except Exception as e:
        print(f"[backend.model_trainer] 裁剪音频异常: {e}")
        return None

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
                   (os.path.exists(video_path) and os.path.getmtime(video_path) > os.path.getmtime(target_video_path)):
                    shutil.copy2(video_path, target_video_path)
                    print(f"[backend.model_trainer] 已复制视频: {video_path} -> {target_video_path}")
                else:
                    print(f"[backend.model_trainer] 目标视频已存在且较新，跳过复制")
                
                # 更新 video_path 为训练目录中的路径（此时 dataset_path 已经是 TalkingGaussian/data/<video_name>）
                video_path = target_video_path
            
            # 统一确定 dataset_path：
            # - 如果前面是上传目录分支，dataset_path 已经在上面设置好了（TalkingGaussian/data/<video_name>）
            # - 否则，根据视频路径推断数据目录
            if 'dataset_path' not in locals():
                # 对于非上传目录的视频：
                #   - 相对简单路径：默认放在 TalkingGaussian/data/<video_name>
                #   - 绝对路径或已经在 TalkingGaussian/ 下：使用其所在目录
                if not os.path.isabs(video_path) and not video_path.startswith('TalkingGaussian/'):
                    dataset_path = os.path.join("TalkingGaussian", "data", video_name)
                else:
                    dataset_path = os.path.dirname(video_path)
            
            # 检查数据是否已预处理（检查 transforms_train.json）
            transforms_file = os.path.join(dataset_path, "transforms_train.json")
            if not os.path.exists(transforms_file):
                print("[backend.model_trainer] 数据未预处理，开始预处理...")
                # 执行数据预处理 - 使用conda run切换到talking_gaussian环境
                # 1. 运行 process.py (基础预处理)
                print("[backend.model_trainer] 步骤 1/3: 运行基础预处理 (process.py)...")
                preprocess_cmd = [
                    'conda', 'run', '-n', 'talking_gaussian', '--no-capture-output',
                    'python', 'TalkingGaussian/data_utils/process.py',
                    video_path,
                    '--asr', audio_extractor
                ]
                
                preprocess_result = subprocess.run(
                    preprocess_cmd,
                    capture_output=False,
                    text=True,
                    cwd='.'
                )
                
                if preprocess_result.returncode != 0:
                    print(f"[backend.model_trainer] 基础预处理失败")
                    return _resp("error", None, f"基础预处理失败，请检查控制台日志")

                # 2. 运行 create_teeth_mask.py (生成牙齿遮罩)
                print("[backend.model_trainer] 步骤 2/3: 生成牙齿遮罩 (create_teeth_mask.py)...")
                # 注意：create_teeth_mask.py 需要 dataset 目录作为参数
                # 切换到 TalkingGaussian 目录运行，以解决相对路径问题
                # dataset_path 是 "TalkingGaussian/data/Macron1"，在 TalkingGaussian 目录下应为 "data/Macron1"
                relative_dataset_path = os.path.relpath(dataset_path, "TalkingGaussian")
                
                teeth_mask_cmd = [
                    'conda', 'run', '-n', 'talking_gaussian', '--no-capture-output',
                    'bash', '-c',
                    f'export PYTHONPATH=data_utils/easyportrait && python data_utils/easyportrait/create_teeth_mask.py {relative_dataset_path}'
                ]

                teeth_mask_result = subprocess.run(
                    teeth_mask_cmd,
                    capture_output=False,
                    text=True,
                    cwd='TalkingGaussian'  # 临时切换工作目录
                )

                if teeth_mask_result.returncode != 0:
                    print(f"[backend.model_trainer] 生成牙齿遮罩失败")
                    return _resp("error", None, f"生成牙齿遮罩失败，请检查控制台日志")

                # 3. 运行 extract_ds_features.py (生成 DeepSpeech 特征)
                # 只有当 audio_extractor 为 deepspeech 时才需要这一步，但为了保险起见，如果 aud_ds.npy 不存在，我们手动生成它
                # 注意：process.py 可能已经生成了 aud.npy，我们需要确保 aud_ds.npy 存在
                print("[backend.model_trainer] 步骤 3/3: 检查并生成 DeepSpeech 特征...")
                
                aud_wav_path = os.path.join(dataset_path, "aud.wav")
                aud_ds_npy_path = os.path.join(dataset_path, "aud_ds.npy")
                
                # 如果 aud_ds.npy 不存在，则运行提取脚本
                if not os.path.exists(aud_ds_npy_path):
                    ds_cmd = [
                        'conda', 'run', '-n', 'talking_gaussian', '--no-capture-output',
                        'python', 'TalkingGaussian/data_utils/deepspeech_features/extract_ds_features.py',
                        '--input', aud_wav_path,
                        '--output', aud_ds_npy_path
                    ]
                    
                    ds_result = subprocess.run(
                        ds_cmd,
                        capture_output=False,
                        text=True,
                        cwd='.'
                    )
                    
                    if ds_result.returncode != 0:
                        print(f"[backend.model_trainer] DeepSpeech 特征提取失败")
                        return _resp("error", None, f"DeepSpeech 特征提取失败，请检查控制台日志")
                else:
                    print(f"[backend.model_trainer] aud_ds.npy 已存在，跳过提取")

                print("[backend.model_trainer] 所有数据预处理完成")

            # 自动修复文件名：如果存在 aud.npy 但不存在 aud_ds.npy，则复制一份
            # TalkingGaussian 训练脚本通常需要 aud_ds.npy 作为 DeepSpeech 特征
            aud_npy_path = os.path.join(dataset_path, "aud.npy")
            aud_ds_npy_path = os.path.join(dataset_path, "aud_ds.npy")
            
            if os.path.exists(aud_npy_path) and not os.path.exists(aud_ds_npy_path):
                print(f"[backend.model_trainer] 检测到 aud.npy，自动复制为 aud_ds.npy 以匹配训练脚本要求...")
                try:
                    shutil.copy2(aud_npy_path, aud_ds_npy_path)
                    print(f"[backend.model_trainer] 文件复制成功: {aud_ds_npy_path}")
                except Exception as e:
                    print(f"[backend.model_trainer] 文件复制失败: {e}")
            
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
            # 修改为不捕获输出，直接打印到控制台
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                env=env,
                cwd='.'
            )
            
            # print("[backend.model_trainer] 训练输出:", result.stdout) # 不再需要打印，因为已经实时输出了
            if result.returncode != 0:
                 print("[backend.model_trainer] 训练失败，请检查上方日志")

            if result.returncode == 0:
                print(f"[backend.model_trainer] 训练完成，模型保存在: {workspace}")
                
                # 方案B：训练完成后，自动复制提取的音频到 static/uploads/audios/
                extracted_audio_path = os.path.join(dataset_path, "aud.wav")
                reference_audio_path = None
                reference_audio_short = None

                if os.path.exists(extracted_audio_path):
                    uploads_audio_dir = "static/uploads/audios"
                    os.makedirs(uploads_audio_dir, exist_ok=True)
                    reference_audio_path = os.path.join(uploads_audio_dir, f"{video_name}_reference.wav")
                    shutil.copy2(extracted_audio_path, reference_audio_path)
                    print(f"[backend.model_trainer] 已复制提取的音频到: {reference_audio_path}")

                    # 生成裁剪版，便于 CosyVoice 等 30s 限制的场景
                    reference_audio_short = os.path.join(uploads_audio_dir, f"{video_name}_reference_short.wav")
                    trimmed = _trim_audio(reference_audio_path, reference_audio_short, max_seconds=29.5)
                    if trimmed:
                        reference_audio_short = trimmed
                        print(f"[backend.model_trainer] 已生成裁剪版参考音频: {reference_audio_short}")
                    else:
                        reference_audio_short = None
                else:
                    print(f"[backend.model_trainer] 警告: 未找到提取的音频文件: {extracted_audio_path}")
                
                # 返回相对路径格式，便于后续使用
                # workspace 是 "TalkingGaussian/output/May"，返回 "output/May"
                relative_workspace = os.path.relpath(workspace, "TalkingGaussian")
                if relative_workspace.startswith('..'):
                    # 如果相对路径计算失败，使用简单方法
                    relative_workspace = workspace.replace('TalkingGaussian/', '').lstrip('/')
                # 训练预览无声视频（训练脚本生成的 out.mp4）
                preview_src = os.path.join(workspace, "test", "ours_None", "renders", "out.mp4")
                preview_video = None
                if os.path.exists(preview_src):
                    preview_dir = os.path.join("static", "videos")
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_name = f"{video_name}_train_preview.mp4"
                    preview_dst = os.path.join(preview_dir, preview_name)
                    try:
                        shutil.copy2(preview_src, preview_dst)
                        preview_video = preview_dst.replace('\\', '/')
                        print(f"[backend.model_trainer] 已复制训练预览视频到: {preview_video}")
                    except Exception as e:
                        print(f"[backend.model_trainer] 复制训练预览视频失败: {e}")

                print(f"[backend.model_trainer] 返回相对路径: {relative_workspace}")
                return _resp("success", relative_workspace, "训练完成", preview_video, reference_audio_path, reference_audio_short)
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
