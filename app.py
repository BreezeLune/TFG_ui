from flask import Flask, render_template, request, jsonify
import os
from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

app = Flask(__name__)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),
            "target_text": request.form.get('target_text'),
        }

        video_path = generate_video(data)
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')


# 模型训练界面
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        data = {
            "model_choice": request.form.get('model_choice'),
            "ref_video": request.form.get('ref_video'),
            "gpu_choice": request.form.get('gpu_choice'),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params')
        }

        video_path = train_model(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('model_training.html')


# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            # 新的语音克隆参数格式
            "voice_clone_type": request.form.get('voice_clone_type'),  # current_recording / preset_voice / custom
            "preset_voice_name": request.form.get('preset_voice_name'),  # 预设音色名称
            "custom_voice_file": request.form.get('custom_voice_file'),  # 自定义音频文件名
            # 保留旧参数以兼容（如果前端还未更新）
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
        }

        video_path = chat_response(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    # 确保目录存在
    os.makedirs('./static/audios', exist_ok=True)
    
    # 保存文件
    audio_file.save('./static/audios/input.wav')
    
    return jsonify({'status': 'success', 'message': '音频保存成功'})

@app.route('/upload_voice_clone', methods=['POST'])
def upload_voice_clone():
    """
    上传自定义语音克隆参考音频文件
    用于语音克隆的参考音频上传接口
    """
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    # 检查文件扩展名
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        return jsonify({'status': 'error', 'message': '不支持的音频格式，请上传 .wav, .mp3, .m4a 或 .flac 文件'})
    
    # 确保自定义音频目录存在
    custom_voice_dir = './static/audios/custom_voice'
    os.makedirs(custom_voice_dir, exist_ok=True)
    
    # 生成唯一文件名（使用时间戳避免覆盖）
    import time
    timestamp = int(time.time())
    file_ext = os.path.splitext(audio_file.filename)[1]
    filename = f"custom_voice_{timestamp}{file_ext}"
    filepath = os.path.join(custom_voice_dir, filename)
    
    # 保存文件
    audio_file.save(filepath)
    
    return jsonify({
        'status': 'success', 
        'message': '音频上传成功',
        'filename': filename,
        'filepath': filepath.replace('\\', '/')
    })


if __name__ == '__main__':
    app.run(debug=True, port = 5001)
