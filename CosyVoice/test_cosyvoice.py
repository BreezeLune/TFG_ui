"""
CosyVoice 语音克隆脚本
"""

import sys
import os
import argparse
import torchaudio

sys.path.append('./third_party/Matcha-TTS')
sys.path.append('.')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav



def main():
    parser = argparse.ArgumentParser(description='CosyVoice 语音克隆工具')
    parser.add_argument('--model_dir', 
                        default='./pretrained_models/CosyVoice2-0.5B',
                        help='模型目录路径')
    parser.add_argument('--prompt_wav', 
                        default='./asset/zero_shot_prompt.wav',
                        help='语音源文件路径')
    parser.add_argument('--prompt_text',
                        default='希望你以后能够做的比我还好呦。',
                        help='语音源文件对应文本')
    parser.add_argument('--tts_text',
                        default='Hello! This is a voice cloning test using CosyVoice. \
                        This model can clone voices with high fidelity.\
                        Let\'s see how well it performs! \
                        This is an apple. I like apples. Apples are good for our health.\
                        Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance.\
                        Thank you for listening. Goodbye! ',
                        help='要合成的文本内容')
    parser.add_argument('--language',
                        choices=['zh', 'en'],
                        default='en',
                        help='输出语言类型: zh (中文), en (英文)')
    parser.add_argument('--speed',
                        type=float,
                        default=1.0,
                        help='语速调节 (0.5-2.0)，1.0为正常速度')
    parser.add_argument('--output_file',
                        default='cloned_voice.wav',
                        help='输出文件名')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = 'test_result'
    os.makedirs(output_dir, exist_ok=True)

    # 检查语音源文件是否存在
    if not os.path.exists(args.prompt_wav):
        print(f"错误: 语音源文件 {args.prompt_wav} 不存在")
        return

    # 加载模型
    print("正在加载模型...")
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # 加载参考音频
    print("正在加载参考音频...")
    prompt_speech_16k = load_wav(args.prompt_wav, 16000)

    # 根据语言选项处理文本和选择方法
    tts_text = args.tts_text
    output_filepath = os.path.join(output_dir, args.output_file)
    
    print(f"正在合成语音: {tts_text}")
    
    # 验证语速参数范围
    if args.speed < 0.5 or args.speed > 2.0:
        print(f"警告: speed={args.speed} 不在有效范围 [0.5, 2.0] 内，使用默认值 1.0")
        args.speed = 1.0
    
    # 执行语音克隆
    output_files = []
    if args.language == 'zh':
        # 中文使用 inference_zero_shot 方法
        for i, j in enumerate(cosyvoice.inference_zero_shot(
            tts_text,
            args.prompt_text,
            prompt_speech_16k,
            stream=False,
            speed=args.speed)):
            
            output_filename = f'{os.path.splitext(output_filepath)[0]}_{i}.wav'
            torchaudio.save(output_filename, j['tts_speech'], cosyvoice.sample_rate)
            output_files.append(output_filename)
            print(f"已保存: {output_filename}")
    else:
        # 英文使用 inference_cross_lingual 方法，添加语言标记
        tts_text_with_tag = f'<|en|>{tts_text}'
        for i, j in enumerate(cosyvoice.inference_cross_lingual(
            tts_text_with_tag,
            prompt_speech_16k,
            stream=False,
            speed=args.speed)):
            
            output_filename = f'{os.path.splitext(output_filepath)[0]}_{i}.wav'
            torchaudio.save(output_filename, j['tts_speech'], cosyvoice.sample_rate)
            output_files.append(output_filename)
            print(f"已保存: {output_filename}")
    
    if output_files:
        print(f"语音克隆完成，共生成 {len(output_files)} 个文件:")
        for f in output_files:
            print(f"  - {f}")
    else:
        print("语音克隆失败，未生成任何文件")


if __name__ == '__main__':
    main()