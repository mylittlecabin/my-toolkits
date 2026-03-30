"""
========================================
批量视频转文字工具
========================================

功能描述：
本脚本基于 FunASR 和 FFmpeg，支持批量将指定目录下的视频文件转换为纯文本 以及 SRT 字幕文件。

⚠️ 前置依赖 (非常重要，请务必确认已安装)：
1. FFmpeg: 必须安装在系统中，且能在终端直接运行 ffmpeg 命令。
   - macOS: brew install ffmpeg
   - Windows: 需下载并手动添加到系统 Path 环境变量
2. Python 依赖库: 
   - pip install funasr torch torchaudio

使用说明与范例：
----------------
基础用法 (处理目录下所有默认的 mp4 文件):
    python batch_video2text.py <视频目录> <输出目录>
    范例: python batch_video2text.py ./my_videos ./my_texts

进阶用法 (处理其他格式，如 .mkv):
    python batch_video2text.py <视频目录> <输出目录> --ext .mkv
    范例: python batch_video2text.py ./my_videos ./my_texts --ext .mkv

开启 GPU 加速 (需要有 NVIDIA 显卡且安装了 CUDA 版本的 PyTorch):
    python batch_video2text.py <视频目录> <输出目录> --gpu
    范例: python batch_video2text.py ./my_videos ./my_texts --gpu
"""

import os
import subprocess
import glob
import argparse
from funasr import AutoModel

# ================= 功能函数区域 =================

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_audio_ffmpeg(video_path, audio_path):
    """
    使用 ffmpeg 将视频转换为音频 (WAV格式, 16kHz)
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", audio_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 音频提取失败: {video_path}")
        print(e.stderr.decode())
        return False

def format_time(ms):
    """将毫秒转换为 SRT 时间格式"""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def process_one_video(model, video_path, output_dir):
    """处理单个视频文件"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_audio = os.path.join(output_dir, f"{base_name}_temp.wav")
    
    print(f"\n{'='*20}\n开始处理: {base_name}\n{'='*20}")

    # 1. 视频转音频
    print(">>> [1/3] 提取音频...")
    if not extract_audio_ffmpeg(video_path, temp_audio):
        return

    # 2. 语音识别
    print(">>> [2/3] 语音识别中...")
    try:
        result = model.generate(input=temp_audio)
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        return

    # 3. 保存结果
    print(">>> [3/3] 保存文件...")
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
            import re
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
    
    # 清理临时文件
    if os.path.exists(temp_audio):
        os.remove(temp_audio)

def main():
    # 1. 解析命令行参数 (增加了 epilog 来显示范例)
    parser = argparse.ArgumentParser(
        description="批量视频转文字工具 (FunASR + FFmpeg)。支持输出 TXT 和 SRT 格式。",
        epilog="""
使用范例:
  1. 基础用法 (处理 mp4):
     python %(prog)s ./my_videos ./my_texts
  2. 处理其他格式 (如 .mkv):
     python %(prog)s ./my_videos ./my_texts --ext .mkv
  3. 使用 GPU 加速:
     python %(prog)s ./my_videos ./my_texts --gpu
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", help="视频文件所在的目录路径")
    parser.add_argument("output_dir", help="结果文本输出的目录路径")
    parser.add_argument("--ext", default=".mp4", help="视频文件后缀名 (默认: .mp4)")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU加速 (默认不使用)")
    args = parser.parse_args()

    # 2. 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"❌ 错误：输入目录不存在 -> {args.input_dir}")
        return

    ensure_dir(args.output_dir)

    # 查找文件
    search_pattern = os.path.join(args.input_dir, f"*{args.ext}")
    video_files = glob.glob(search_pattern)
    
    if not video_files:
        print(f"⚠️ 在 {args.input_dir} 下没有找到 {args.ext} 文件。")
        return

    print(f"找到 {len(video_files)} 个视频文件。")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    print(f"设备模式: {'GPU' if args.gpu else 'CPU'}")

    # 3. 初始化模型
    print("正在初始化模型...")
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        device="cuda" if args.gpu else "cpu"
    )
    
    # 4. 批量处理
    for video_path in video_files:
        try:
            process_one_video(model, video_path, args.output_dir)
        except Exception as e:
            print(f"❌ 处理出错 {video_path}: {e}")
            continue

    print("\n🎉 全部任务完成！")

if __name__ == "__main__":
    main()

