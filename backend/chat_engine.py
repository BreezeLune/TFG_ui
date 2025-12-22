import os
import subprocess
import shutil
import speech_recognition as sr
from zhipuai import ZhipuAI
from backend.llm_service import query_llm 
# 预设音色配置
PRESET_VOICES = {
    "default": "./CosyVoice/asset/zero_shot_prompt.wav",
    "cross_lingual": "./CosyVoice/asset/cross_lingual_prompt.wav",
    # 可以继续添加更多预设音色
    # "voice1": "./CosyVoice/asset/voice1.wav",
    # "voice2": "./CosyVoice/asset/voice2.wav",
}

def get_voice_clone_reference(voice_clone_type, preset_voice_name=None, custom_voice_file=None, fallback_voice_clone=None):
    """
    根据选择类型获取语音克隆参考音频路径
    
    Args:
        voice_clone_type: 选择类型
            - "current_recording": 使用当前录音
            - "preset_voice": 使用预设音色
            - "custom": 使用自定义上传的音频
        preset_voice_name: 预设音色名称（当 voice_clone_type 为 "preset_voice" 时使用）
        custom_voice_file: 自定义音频文件名（当 voice_clone_type 为 "custom" 时使用）
        fallback_voice_clone: 兼容旧版本的参数（如果提供了，优先使用）
    
    Returns:
        参考音频文件路径
    """
    # 兼容旧版本：如果提供了 voice_clone 参数，直接使用
    if fallback_voice_clone:
        if os.path.exists(fallback_voice_clone):
            print(f"[backend.chat_engine] 使用兼容模式，参考音频: {fallback_voice_clone}")
            return fallback_voice_clone
        else:
            print(f"[backend.chat_engine] 兼容模式路径不存在，使用默认: {fallback_voice_clone}")
    
    # 根据类型选择参考音频
    if voice_clone_type == "current_recording":
        # 使用当前录音
        reference_path = "./static/audios/input.wav"
        if os.path.exists(reference_path):
            print(f"[backend.chat_engine] 使用当前录音作为参考音频: {reference_path}")
            return reference_path
        else:
            print(f"[backend.chat_engine] 当前录音不存在，使用默认预设音色")
            return PRESET_VOICES.get("default", "./CosyVoice/asset/zero_shot_prompt.wav")
    
    elif voice_clone_type == "preset_voice":
        # 使用预设音色
        if preset_voice_name and preset_voice_name in PRESET_VOICES:
            reference_path = PRESET_VOICES[preset_voice_name]
            if os.path.exists(reference_path):
                print(f"[backend.chat_engine] 使用预设音色: {preset_voice_name} -> {reference_path}")
                return reference_path
            else:
                print(f"[backend.chat_engine] 预设音色文件不存在: {reference_path}，使用默认")
        else:
            print(f"[backend.chat_engine] 未找到预设音色: {preset_voice_name}，使用默认")
        # 使用默认预设音色
        return PRESET_VOICES.get("default", "./CosyVoice/asset/zero_shot_prompt.wav")
    
    elif voice_clone_type == "custom":
        # 使用自定义上传的音频
        if custom_voice_file:
            reference_path = f"./static/audios/custom_voice/{custom_voice_file}"
            if os.path.exists(reference_path):
                print(f"[backend.chat_engine] 使用自定义音频: {reference_path}")
                return reference_path
            else:
                print(f"[backend.chat_engine] 自定义音频文件不存在: {reference_path}，使用默认预设音色")
        else:
            print(f"[backend.chat_engine] 未提供自定义音频文件名，使用默认预设音色")
        # 使用默认预设音色
        return PRESET_VOICES.get("default", "./CosyVoice/asset/zero_shot_prompt.wav")
    
    else:
        # 默认使用预设音色
        print(f"[backend.chat_engine] 未知的语音克隆类型: {voice_clone_type}，使用默认预设音色")
        return PRESET_VOICES.get("default", "./CosyVoice/asset/zero_shot_prompt.wav")

def chat_response(data):
    """
    完整的实时对话系统视频生成逻辑。
    流程：ASR → LLM → CosyVoice语音克隆 → TalkingGaussian视频生成
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    try:
        # 步骤1：语音识别（ASR）
        input_audio = "./static/audios/input.wav"
        input_text = "./static/text/input.txt"
        os.makedirs("./static/text", exist_ok=True)
        
        if not os.path.exists(input_audio):
            print(f"[backend.chat_engine] 音频文件不存在: {input_audio}")
            return os.path.join("static", "videos", "chat_response.mp4")
        
        recognized_text = audio_to_text(input_audio, input_text)
        if not recognized_text:
            print("[backend.chat_engine] 语音识别失败")
            return os.path.join("static", "videos", "chat_response.mp4")
        
        # 步骤2：大模型生成回复
        output_text = "./static/text/output.txt"
        # api_key = os.getenv('ZHIPU_API_KEY', '31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M')
        # model = "glm-4-plus"
        # reply_text = get_ai_response(input_text, output_text, api_key, model)
        api_choice = data.get('api_choice', 'Zhipu AI')
        reply_text = query_llm(recognized_text, api_choice)
        # 将回复保存到文件（保持原有逻辑的副作用）
        with open(output_text, 'w', encoding='utf-8') as f:
            f.write(reply_text)
        
        if not reply_text:
            print("[backend.chat_engine] LLM回复生成失败")
            return os.path.join("static", "videos", "chat_response.mp4")
        
        # 步骤3：语音克隆（CosyVoice）
        # 获取参考音频路径（支持新的三种选择方式）
        voice_clone_type = data.get('voice_clone_type')
        preset_voice_name = data.get('preset_voice_name')
        custom_voice_file = data.get('custom_voice_file')
        fallback_voice_clone = data.get('voice_clone')  # 兼容旧版本
        
        voice_clone_ref = get_voice_clone_reference(
            voice_clone_type=voice_clone_type,
            preset_voice_name=preset_voice_name,
            custom_voice_file=custom_voice_file,
            fallback_voice_clone=fallback_voice_clone
        )
        
        # 最终检查文件是否存在
        if not os.path.exists(voice_clone_ref):
            print(f"[backend.chat_engine] 警告：参考音频文件不存在: {voice_clone_ref}")
            # 尝试使用默认路径
            default_ref = './CosyVoice/asset/zero_shot_prompt.wav'
            if os.path.exists(default_ref):
                print(f"[backend.chat_engine] 使用默认参考音频: {default_ref}")
                voice_clone_ref = default_ref
            else:
                print(f"[backend.chat_engine] 错误：默认参考音频也不存在: {default_ref}")
                return os.path.join("static", "videos", "chat_response.mp4")
        
        # 获取 CosyVoice 参数
        cosyvoice_params = data.get('cosyvoice_params', {})
        language = cosyvoice_params.get('language', 'zh')  # 语言类型：zh 或 en
        speed = cosyvoice_params.get('speed', 1.0)  # 方案一：语速调节，默认 1.0
        
        # 验证语言类型
        if language not in ['zh', 'en']:
            print(f"[backend.chat_engine] 警告：language={language} 不在有效范围 ['zh','en'] 内，使用默认值 'zh'")
            language = 'zh'
        
        # 验证语速参数
        try:
            speed = float(speed)
            if speed < 0.5 or speed > 2.0:
                print(f"[backend.chat_engine] 警告：speed={speed} 不在有效范围 [0.5, 2.0] 内，使用默认值 1.0")
                speed = 1.0
        except (ValueError, TypeError):
            print(f"[backend.chat_engine] 警告：speed={speed} 不是有效数字，使用默认值 1.0")
            speed = 1.0
        
        # 生成克隆音频
        os.makedirs("./static/audios", exist_ok=True)
        tts_output = "./static/audios/tts_output.wav"
        
        cloned_audio = text_to_speech_cosyvoice(
            text=reply_text,
            prompt_wav=voice_clone_ref,
            output_file=tts_output,
            language=language,  # 使用前端选择的语言类型
            speed=speed  # 方案一：语速调节
        )
        
        if not cloned_audio or not os.path.exists(cloned_audio):
            print("[backend.chat_engine] 语音克隆失败")
            return os.path.join("static", "videos", "chat_response.mp4")
        
        # 步骤4：TalkingGaussian生成视频
        # 使用相对路径格式（统一路径格式）
        model_param = data.get('model_param', 'output/talking_May')  # 改为相对路径
        dataset_path = data.get('dataset_path', 'data/May')  # 改为相对路径
        gpu_choice = data.get('gpu_choice', 'GPU0')
        audio_extractor = data.get('audio_extractor', 'deepspeech')
        
        # 获取推理参数（方案二：渲染细节等级）
        inference_params = data.get('inference_params', {})
        sh_degree = inference_params.get('sh_degree', 2)  # 默认值 2（标准模式）
        # 验证 sh_degree 范围
        if sh_degree not in [0, 1, 2, 3]:
            print(f"[backend.chat_engine] 警告：sh_degree={sh_degree} 不在有效范围 [0,1,2,3] 内，使用默认值 2")
            sh_degree = 2
        
        # 调用视频生成
        video_data = {
            'model_name': 'TalkingGaussian',
            'model_param': model_param,
            'ref_audio': cloned_audio,
            'dataset_path': dataset_path,
            'gpu_choice': gpu_choice,
            'audio_extractor': audio_extractor,
            'inference_params': {
                'sh_degree': sh_degree  # 渲染细节等级（方案二）
            }
        }
        
        # 导入video_generator模块
        from backend.video_generator import generate_video
        video_path = generate_video(video_data)
        
        print(f"[backend.chat_engine] 完整对话流程完成，视频路径：{video_path}")
        return video_path
        
    except Exception as e:
        print(f"[backend.chat_engine] 错误: {e}")
        import traceback
        traceback.print_exc()
        return os.path.join("static", "videos", "chat_response.mp4")

def audio_to_text(input_audio, input_text):
    """
    语音识别（ASR）
    优先使用Google语音识别，如果网络不可用则使用fallback文本
    """
    try:
        # 初始化识别器
        recognizer = sr.Recognizer()
        
        # 加载音频文件
        with sr.AudioFile(input_audio) as source:
            # 调整环境噪声
            recognizer.adjust_for_ambient_noise(source)
            # 读取音频数据
            audio_data = recognizer.record(source)
            
            print("正在识别语音...")
            
            # 尝试使用Google语音识别
            try:
                text = recognizer.recognize_google(audio_data, language='zh-CN')
                print(f"✅ Google语音识别成功")
            except sr.RequestError as e:
                print(f"⚠️  Google语音识别服务不可用: {e}")
                print("💡 使用fallback文本（测试模式）")
                # 使用fallback文本（用于测试）
                text = "你好，这是一个测试。请告诉我你能做什么？"
                print(f"📝 Fallback文本: {text}")
            except sr.UnknownValueError:
                print("⚠️  无法识别音频内容，使用fallback文本")
                text = "你好，这是一个测试。请告诉我你能做什么？"
            
            # 将结果写入文件
            with open(input_text, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"语音识别完成！结果已保存到: {input_text}")
            print(f"识别结果: {text}")
            
            return text
            
    except FileNotFoundError:
        print(f"音频文件不存在: {input_audio}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_ai_response(input_text, output_text, api_key, model):
    try:
        client = ZhipuAI(api_key=api_key)
        with open(input_text, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}]
        )
        output = response.choices[0].message.content

        with open(output_text, 'w', encoding='utf-8') as file:
            file.write(output)

        print(f"答复已保存到: {output_text}")
        return output
    except Exception as e:
        print(f"[backend.chat_engine] LLM调用失败: {e}")
        return None

def text_to_speech_cosyvoice(text, prompt_wav, output_file, language='zh', model_dir=None, speed=1.0):
    """
    使用CosyVoice进行语音克隆
    
    Args:
        text: 要合成的文本
        prompt_wav: 参考音频路径
        output_file: 输出音频文件路径
        language: 语言类型 ('zh' 或 'en')
        model_dir: CosyVoice模型目录（可选）
        speed: 语速调节 (0.5-2.0)，1.0为正常速度（方案一：语速调节）
    
    Returns:
        生成的音频文件路径，失败返回None
    """
    try:
        # 默认模型目录
        if model_dir is None:
            model_dir = './CosyVoice/pretrained_models/CosyVoice2-0.5B'
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            print(f"[backend.chat_engine] CosyVoice模型目录不存在: {model_dir}")
            return None
        
        # 检查参考音频是否存在
        if not os.path.exists(prompt_wav):
            print(f"[backend.chat_engine] 参考音频文件不存在: {prompt_wav}")
            return None
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 构建CosyVoice调用命令
        # 使用Shell脚本封装，脚本内部会使用conda run切换到cosyvoice环境
        cosyvoice_script = './CosyVoice/run_cosyvoice.sh'
        
        if not os.path.exists(cosyvoice_script):
            print(f"[backend.chat_engine] 找不到CosyVoice脚本: {cosyvoice_script}")
            return None
        
        # 验证 speed 参数范围
        if speed < 0.5 or speed > 2.0:
            print(f"[backend.chat_engine] 警告：speed={speed} 不在有效范围 [0.5, 2.0] 内，使用默认值 1.0")
            speed = 1.0
        
        # 构建命令 - 调用Shell脚本，脚本内部会使用conda run
        cmd = [
            'bash', cosyvoice_script,
            '--model_dir', model_dir,
            '--prompt_wav', prompt_wav,
            '--prompt_text', text[:50],  # 使用文本前50字符作为prompt_text（简化处理）
            '--tts_text', text,
            '--language', language,
            '--speed', str(speed),  # 方案一：语速调节
            '--output_file', os.path.basename(output_file)
        ]
        
        print(f"[backend.chat_engine] 执行CosyVoice命令: {' '.join(cmd)}")
        
        # 执行命令 - 从项目根目录执行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        print("CosyVoice标准输出:", result.stdout)
        if result.stderr:
            print("CosyVoice标准错误:", result.stderr)
        
        if result.returncode != 0:
            print(f"[backend.chat_engine] CosyVoice执行失败，退出码: {result.returncode}")
            return None
        
        # 查找生成的音频文件
        # test_cosyvoice.py 输出到 test_result/ 目录
        output_dir = os.path.join(os.path.dirname(cosyvoice_script), 'test_result')
        if not os.path.exists(output_dir):
            output_dir = 'CosyVoice/test_result'
        
        # 查找生成的音频文件（可能带索引）
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        generated_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.startswith(base_name) and f.endswith('.wav'):
                    generated_files.append(os.path.join(output_dir, f))
        
        if generated_files:
            # 使用最新的文件
            latest_file = max(generated_files, key=lambda f: os.path.getctime(f))
            # 复制到目标位置
            shutil.copy(latest_file, output_file)
            print(f"[backend.chat_engine] 语音克隆完成: {output_file}")
            return output_file
        else:
            print(f"[backend.chat_engine] 未找到生成的音频文件")
            return None
            
    except Exception as e:
        print(f"[backend.chat_engine] 语音克隆错误: {e}")
        import traceback
        traceback.print_exc()
        return None