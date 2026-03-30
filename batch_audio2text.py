"""
========================================
批量音频转文字工具
========================================

功能描述：
基于 FunASR，批量将指定目录下的音频文件转换为纯文本 以及 SRT 字幕文件。
建议输入由 batch_video2audio.py 生成的 16kHz WAV 文件以获得最佳效果。

⚠️ 前置依赖:
1. Python 依赖库: pip install funasr torch torchaudio
2. 模型文件: 首次运行会自动从 ModelScope 下载模型(约 1-2GB)，请确保网络通畅。

使用说明与范例：
----------------
基础用法 (处理目录下所有默认的 wav 文件):
    python batch_audio2text.py <音频目录> <文本输出目录>
    范例: python batch_audio2text.py ./my_audios ./my_texts

进阶用法 (处理其他格式，如 .mp3):
    python batch_audio2text.py <音频目录> <文本输出目录> --ext .mp3
    范例: python batch_audio2text.py ./my_audios ./my_texts --ext .mp3

开启 GPU 加速 (需要有 NVIDIA 显卡且安装了 CUDA 版本的 PyTorch):
    python batch_audio2text.py <音频目录> <文本输出目录> --gpu
    范例: python batch_audio2text.py ./my_audios ./my_texts --gpu
"""

import os
import glob
import argparse
import re
from funasr import AutoModel

# ================= 功能函数区域 =================

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_time(ms):
    """将毫秒转换为 SRT 时间格式"""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def process_one_audio(model, audio_path, output_dir):
    """处理单个音频文件"""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    print(f"\n{'='*20}\n开始处理: {base_name}\n{'='*20}")

    # 1. 语音识别
    print(">>> [1/2] 语音识别中...")
    try:
        result = model.generate(input=audio_path)
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        return

    # 2. 保存结果
    print(">>> [2/2] 保存文件...")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    srt_path = os.path.join(output_dir, f"{base_name}.srt")
    
    res_data = result[0]
    full_text = res_data.get("text", "")
    timestamps = res_data.get("timestamp", [])

    # 保存 TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # 保存 SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        if timestamps:
            sentences = re.split(r'([，。！？、])', full_text)
            sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
            
            char_idx = 0
            srt_index = 1
            for sent in sentences:
                if not sent.strip():
                    continue
                
                num_chars = len(sent)
                start_time = timestamps[char_idx][0] if char_idx < len(timestamps) else 0
                end_idx = char_idx + num_chars - 1
                end_time = timestamps[end_idx][1] if end_idx < len(timestamps) else start_time
                
                f.write(f"{srt_index}\n")
                f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                f.write(f"{sent}\n\n")
                
                srt_index += 1
                char_idx += num_chars
        else:
            f.write(f"1\n00:00:00,000 --> 10:00:00,000\n{full_text}\n")

    print(f"✅ 完成！已保存: {base_name}.txt/.srt")

def main():
    parser = argparse.ArgumentParser(
        description="批量音频转文字工具",
        epilog="""
使用范例:
  1. 基础用法:
     python %(prog)s ./my_audios ./my_texts
  2. 处理 mp3 格式:
     python %(prog)s ./my_audios ./my_texts --ext .mp3
  3. 使用 GPU 加速:
     python %(prog)s ./my_audios ./my_texts --gpu
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", help="音频文件所在的目录路径")
    parser.add_argument("output_dir", help="结果文本输出的目录路径")
    parser.add_argument("--ext", default=".wav", help="音频文件后缀名 (默认: .wav)")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU加速 (默认不使用)")
    args = parser.parse_args()

    # 检查目录
    if not os.path.isdir(args.input_dir):
        print(f"❌ 错误：输入目录不存在 -> {args.input_dir}")
        return

    ensure_dir(args.output_dir)

    # 查找文件
    search_pattern = os.path.join(args.input_dir, f"*{args.ext}")
    audio_files = glob.glob(search_pattern)
    
    if not audio_files:
        print(f"⚠️ 在 {args.input_dir} 下没有找到 {args.ext} 文件。")
        return

    print(f"找到 {len(audio_files)} 个音频文件。")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    print(f"设备模式: {'GPU' if args.gpu else 'CPU'}")

    # 初始化模型
    print("正在初始化 FunASR 模型...")
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        device="cuda" if args.gpu else "cpu"
    )
    
    # 批量处理
    for audio_path in audio_files:
        try:
            process_one_audio(model, audio_path, args.output_dir)
        except Exception as e:
            print(f"❌ 处理出错 {audio_path}: {e}")
            continue

    print("\n🎉 全部任务完成！")

if __name__ == "__main__":
    main()

