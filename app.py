from flask import Flask, render_template, request, jsonify
import os
import threading
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
        # 获取推理参数（渲染细节等级）
        sh_degree = request.form.get('sh_degree')
        inference_params = {}
        if sh_degree:
            try:
                sh_degree_int = int(sh_degree)
                if sh_degree_int in [0, 1, 2, 3]:
                    inference_params['sh_degree'] = sh_degree_int
                else:
                    print(f"[app] 警告：sh_degree={sh_degree_int} 不在有效范围 [0,1,2,3] 内，使用默认值")
            except ValueError:
                print(f"[app] 警告：sh_degree={sh_degree} 不是有效整数，使用默认值")
        
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),
            "target_text": request.form.get('target_text'),
            "inference_params": inference_params if inference_params else {}  # 方案二：渲染细节等级
        }

        video_path = generate_video(data)
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')


# 训练任务状态存储（简单实现，生产环境应使用Redis等）
training_tasks = {}

def train_model_async(data, task_id):
    """异步执行训练任务"""
    try:
        result = train_model(data)
        if isinstance(result, dict):
            training_tasks[task_id] = {
                'status': result.get('status', 'error'),
                'model_path': result.get('model_path'),
                'message': result.get('message', '')
            }
        else:
            # 兼容旧返回：只返回路径则认为成功
            training_tasks[task_id] = {
                'status': 'success',
                'model_path': result,
                'message': f'训练完成，模型路径：{result}'
            }
    except Exception as e:
        training_tasks[task_id] = {
            'status': 'error',
            'message': f'训练失败：{str(e)}'
        }

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

        # 生成任务ID
        import time
        task_id = f"train_{int(time.time())}"
        
        # 启动异步训练任务
        thread = threading.Thread(target=train_model_async, args=(data, task_id))
        thread.daemon = True
        thread.start()
        
        # 立即返回，告诉前端训练已开始
        return jsonify({
            'status': 'started',
            'task_id': task_id,
            'message': '训练已开始，请稍候...'
        })

    return render_template('model_training.html')

# 训练状态查询接口
@app.route('/training_status/<task_id>', methods=['GET'])
def training_status(task_id):
    """查询训练任务状态"""
    if task_id in training_tasks:
        return jsonify(training_tasks[task_id])
    else:
        return jsonify({
            'status': 'running',
            'message': '训练进行中...'
        })


# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        # 获取 CosyVoice 参数（语言类型和语速）
        language = request.form.get('language', 'zh')
        speed = request.form.get('speed', '1.0')
        
        # 验证和格式化 speed 参数
        try:
            speed_float = float(speed)
            if speed_float < 0.5 or speed_float > 2.0:
                print(f"[app] 警告：speed={speed_float} 不在有效范围 [0.5, 2.0] 内，使用默认值 1.0")
                speed_float = 1.0
        except ValueError:
            print(f"[app] 警告：speed={speed} 不是有效数字，使用默认值 1.0")
            speed_float = 1.0
        
        cosyvoice_params = {
            'language': language if language in ['zh', 'en'] else 'zh',
            'speed': speed_float  # 方案一：语速调节
        }
        
        # 获取推理参数（方案二：渲染细节等级）
        sh_degree = request.form.get('sh_degree')
        inference_params = {}
        if sh_degree:
            try:
                sh_degree_int = int(sh_degree)
                if sh_degree_int in [0, 1, 2, 3]:
                    inference_params['sh_degree'] = sh_degree_int
            except ValueError:
                print(f"[app] 警告：sh_degree={sh_degree} 不是有效整数，使用默认值")
        
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            # 新的语音克隆参数格式
            "voice_clone_type": request.form.get('voice_clone_type'),  # current_recording / preset_voice / custom
            "preset_voice_name": request.form.get('preset_voice_name'),  # 预设音色名称
            "custom_voice_file": request.form.get('custom_voice_file'),  # 自定义音频文件名（上传后返回的文件名）
            "custom_voice_path": request.form.get('custom_voice_path'),  # 自定义音频完整路径（手动输入）
            # 保留旧参数以兼容（如果前端还未更新）
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
            # CosyVoice 参数
            "cosyvoice_params": cosyvoice_params,  # 语言类型选择
            # TalkingGaussian 推理参数
            "inference_params": inference_params if inference_params else {}  # 方案二：渲染细节等级
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
    
    # 保存文件（固定名称，供后续流程使用）
    input_path = './static/audios/input.wav'
    audio_file.save(input_path)
    
    # 同时保存一个带时间戳的副本（用于历史记录）
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    timestamped_path = f'./static/audios/input_{timestamp}.wav'
    import shutil
    shutil.copy(input_path, timestamped_path)
    
    return jsonify({
        'status': 'success', 
        'message': '音频保存成功',
        'timestamped_path': timestamped_path.replace('\\', '/')
    })

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


@app.route('/upload_au', methods=['POST'])
def upload_au():
    """
    上传 AU 特征文件（au.csv）
    用途：当已在本地/其它环境生成 au.csv 时，直接上传到 TalkingGaussian/data/<ID>/ 目录。
    请求字段：
      - project_id: 必填，训练数据的 ID（对应目录名，如 May）
      - au_file: 必填，au.csv 文件
    """
    project_id = request.form.get('project_id')
    if not project_id:
        return jsonify({'status': 'error', 'message': '缺少 project_id'})

    if 'au_file' not in request.files:
        return jsonify({'status': 'error', 'message': '没有上传文件字段 au_file'})

    au_file = request.files['au_file']
    if au_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    # 仅允许 .csv
    if not au_file.filename.lower().endswith('.csv'):
        return jsonify({'status': 'error', 'message': '仅支持 .csv 文件'})

    # 目标路径：TalkingGaussian/data/<ID>/au.csv
    target_dir = os.path.join('TalkingGaussian', 'data', project_id)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'au.csv')

    au_file.save(target_path)

    return jsonify({
        'status': 'success',
        'message': f'AU 文件已上传到 {target_path}',
        'project_id': project_id,
        'path': target_path.replace('\\', '/')
    })


if __name__ == '__main__':
    # host='0.0.0.0' 允许从外部访问（华为云服务器需要）
    # debug=True 开发模式，生产环境应设置为False
    app.run(debug=True, host='0.0.0.0', port=5001)
