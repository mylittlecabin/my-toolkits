"""
========================================
批量视频转音频工具
========================================

功能描述：
基于 FFmpeg，批量将指定目录下的视频文件提取为无压缩的 WAV 音频文件 (16kHz 单声道)，供后续语音识别使用。

⚠️ 前置依赖:
必须安装 FFmpeg，且能在终端直接运行 ffmpeg 命令。
  - macOS: brew install ffmpeg
  - Windows: 需下载并手动添加到系统 Path 环境变量

使用说明与范例：
----------------
基础用法 (处理目录下所有默认的 mp4 文件):
    python batch_video2audio.py <视频目录> <音频输出目录>
    范例: python batch_video2audio.py ./my_videos ./my_audios

进阶用法 (处理其他格式，如 .mkv):
    python batch_video2audio.py <视频目录> <音频输出目录> --ext .mkv
    范例: python batch_video2audio.py ./my_videos ./my_audios --ext .mkv
"""

import os
import glob
import argparse
import subprocess

# ================= 功能函数区域 =================

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_audio_ffmpeg(video_path, audio_path):
    """使用 ffmpeg 将视频转换为音频 (WAV格式, 16kHz, 单声道)"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", audio_path
    ]
    try:
        # 隐藏常规输出，只在出错时打印错误信息
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {os.path.basename(video_path)}")
        print(f"   错误信息: {e.stderr.decode()[:200]}")
        return False
    except FileNotFoundError:
        print("❌ 致命错误: 未找到 ffmpeg 命令，请确认已安装 FFmpeg 并配置到环境变量中。")
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="批量视频转音频工具 (基于 FFmpeg)",
        epilog="""
使用范例:
  1. 基础用法:
     python %(prog)s ./my_videos ./my_audios
  2. 处理 mkv 格式:
     python %(prog)s ./my_videos ./my_audios --ext .mkv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", help="视频文件所在的目录路径")
    parser.add_argument("output_dir", help="提取出的音频文件存放目录")
    parser.add_argument("--ext", default=".mp4", help="视频文件后缀名 (默认: .mp4)")
    args = parser.parse_args()

    # 检查目录
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

    print(f"找到 {len(video_files)} 个视频文件，开始提取音频...")
    print(f"输出目录: {os.path.abspath(args.output_dir)}\n")

    success_count = 0
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
        
        print(f"处理中: {base_name}{args.ext} -> {base_name}.wav", end="\r")
        if extract_audio_ffmpeg(video_path, audio_path):
            success_count += 1

    print(f"\n\n🎉 全部任务完成！成功提取 {success_count}/{len(video_files)} 个音频文件。")

if __name__ == "__main__":
    main()

