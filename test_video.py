
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip  # 唯一正确导入路径
from model import NewGenerator
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, AudioFileClip
import subprocess
import time
from utils import *
import argparse

def generate_lr_video(
        input_video_path: str,
        output_folder: str = "outputs",
        output_video_name: str = "lr_output.mp4",
        up_scale: int = 2,
        blur_kernel: int = 9,
        blur_sigma: float = 14,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR
):

    # ------------------------- 路径初始化 -------------------------
    input_path = Path(input_video_path).resolve()
    output_dir = Path(output_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_folder = output_dir / "lr_frames"
    frame_folder.mkdir(exist_ok=True)
    output_video_path = output_dir / output_video_name

    # ------------------------- 视频参数获取 -------------------------
    video_cap = cv2.VideoCapture(str(input_path))
    fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lr_width = orig_width // up_scale
    lr_height = orig_height // up_scale

    # ------------------------- 帧处理管道 -------------------------
    to_pilimage = transforms.ToPILImage()
    transform = transforms.Compose([
        transforms.GaussianBlur(blur_kernel, blur_sigma),
        transforms.Resize((lr_height, lr_width), interpolation),
        transforms.ToTensor(),  # 转0-1 Tensor（C×H×W）
        transforms.Lambda(lambda x: x.numpy()),  # 转numpy数组（C×H×W）
        transforms.Lambda(lambda x: x.transpose(1, 2, 0)),  # 转H×W×C（RGB顺序）
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8))  # 归一化到0-255
    ])

    frame_list = []
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_bar = tqdm(range(total_frames), desc="生成低分辨率帧", unit="frame")

    # ------------------------- 逐帧处理 -------------------------
    success, frame = video_cap.read()
    while success and test_bar.n < total_frames:
        # 颜色空间转换（BGR→RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 应用转换管道
        lr_image = to_pilimage(frame_rgb)
        lr_image = defocus_blur(lr_image, 12)#16

        lr_frame = transform(lr_image)
        # 转换为BGR格式以兼容cv2.imwrite
        lr_frame_bgr = cv2.cvtColor(lr_frame, cv2.COLOR_RGB2BGR)

        # 生成帧路径并写入
        frame_filename = f"lr_frame_{test_bar.n:04d}.png"
        frame_path = frame_folder / frame_filename
        write_success = cv2.imwrite(str(frame_path), lr_frame_bgr)

        if not write_success:
            raise RuntimeError(f"帧 {test_bar.n} 写入失败，路径: {frame_path}")

        frame_list.append(str(frame_path))
        success, frame = video_cap.read()
        test_bar.update(1)

    video_cap.release()

    # ------------------------- 音频处理 -------------------------
    temp_audio_path = output_dir / "temp_audio.aac"
    audio_clip = None

    try:
        # 提取音频（固定44100Hz采样率，128k比特率）
        print(f"[INFO] 开始提取音频到 {temp_audio_path}")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vn", "-acodec", "aac", "-ar", "44100", "-ab", "128k",
                str(temp_audio_path)
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg音频提取失败:\n{result.stderr}")
        print(f"[INFO] 音频提取成功")

        # 加载音频并校验
        audio_clip = AudioFileClip(str(temp_audio_path))
        print(
            f"[DEBUG] 音频属性：时长={audio_clip.duration:.2f}s，采样率={audio_clip.fps}Hz，声道数={audio_clip.nchannels}")

        # 同步音频帧率和时长
        if audio_clip.fps != fps:
            audio_clip = audio_clip.set_fps(fps)
        if abs(audio_clip.duration - (total_frames / fps)) > 0.1:
            audio_clip = audio_clip.set_duration(total_frames / fps)
    except Exception as e:
        print(f"[ERROR] 音频处理失败: {e}")
        audio_clip = None

    try:
        frame_list = sorted(frame_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        print(f"[INFO] 开始合并视频，共 {len(frame_list)} 帧")

        image_clip = ImageSequenceClip(frame_list, fps=fps)
        print(f"[INFO] 视频剪辑创建成功，尺寸: {image_clip.size}, 帧率: {image_clip.fps}")

        if audio_clip is not None and getattr(audio_clip, 'duration', 0) > 0:
            final_clip = image_clip.set_audio(audio_clip)
            print("[INFO] 音频有效，开始写入带音频的视频")
        else:
            final_clip = image_clip
            print("[WARNING] 音频无效，生成无声视频")

        # ------------------------- 移除 extra_args，使用旧版参数 -------------------------
        final_clip.write_videofile(
            str(output_video_path),
            codec="libx264",  # 视频编码
            audio_codec="aac",  # 音频编码
            fps=fps,  # 帧率
            verbose=True,  # 显示FFmpeg进度
            # 旧版参数替代方案（直接在参数中指定）
            audio_fps=44100,  # 音频采样率
            threads=4,  # 线程数（提升速度）
            # 以下为旧版FFmpeg参数传递方式（通过字符串拼接）
            # 注意：1.0.3 需将参数合并到 'writer_args' 或直接通过命令行
            # 对于复杂参数，建议使用 subprocess 直接调用FFmpeg（见下方终极方案）
        )
        print(f"[SUCCESS] 视频生成完成: {output_video_path}")

    except Exception as e:
        print(f"[FATAL] 视频合并失败: {e}")
        raise

    finally:
        # 延迟清理临时文件（确保写入完成）
        time.sleep(5)
        for f in frame_folder.glob("*.png"):
            try:
                f.unlink()
            except:
                pass
        if not any(frame_folder.iterdir()):
            frame_folder.rmdir()
        if audio_clip:
            audio_clip.close()
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)

    return str(output_video_path)

def process_crop(images, target_block_width):
    if not images:
        return []
    crop_size = min(images[0].width, images[0].height) // 5
    crops = transforms.FiveCrop(crop_size)(images[0])
    return [np.asarray(img.resize((target_block_width, target_block_width), Image.BICUBIC)) for img in crops]




def generate_sr_compare_video(
        input_lr_video_path: str,
        output_sr_path: str,
        output_compare_path: str,
        model_path: str,
        up_scale: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 路径处理（确保父目录存在）
    output_dir = Path(output_sr_path).parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    lr_frame_folder = temp_dir / "lr_frames"  # 用于存储低分辨率帧的文件夹
    sr_frame_folder = temp_dir / "sr_frames"
    compare_frame_folder = temp_dir / "compare_frames"
    for folder in [lr_frame_folder, sr_frame_folder, compare_frame_folder]:
        folder.mkdir(exist_ok=True)

    # 读取低分辨率视频的帧并保存到文件夹
    video_cap = cv2.VideoCapture(input_lr_video_path)
    if not video_cap.isOpened():
        raise ValueError(f"无法打开输入视频: {input_lr_video_path}")

    fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lr_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    lr_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if lr_width == 0 or lr_height == 0:
        raise ValueError("输入视频尺寸异常，可能文件损坏")

    test_bar = tqdm(total=frame_count, desc="[保存低分辨率帧]")
    success, lr_frame = video_cap.read()
    while success and test_bar.n < frame_count:
        if lr_frame is None:
            break  # 跳过空帧
        # 保存低分辨率帧为图片
        frame_filename = f"lr_frame_{test_bar.n:04d}.png"
        frame_path = lr_frame_folder / frame_filename
        cv2.imwrite(str(frame_path),lr_frame)

        success, lr_frame = video_cap.read()
        test_bar.update(1)

    video_cap.release()

    # 模型加载
    model = NewGenerator(up_scale).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    load_model(model, model_path+"generator", "generator")

    # 预处理/后处理函数
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    def preprocess(x):
        return to_tensor(x).unsqueeze(0).to(device)

    def postprocess(x):
        return (x.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    test_bar = tqdm(total=frame_count, desc="[处理视频帧进行超分辨]")
    # 不转换为字符串，保持Path对象
    lr_frame_list = sorted([x for x in lr_frame_folder.glob("*.png")],
                           key=lambda x: int(x.stem.split("_")[-1]))


    with torch.no_grad():

        for k, (lr_frame_path) in enumerate(zip(lr_frame_list)):
            # 使用Path对象的字符串表示来读取图像
            lr_frame = Image.open(lr_frame_path)
            lr_tensor = preprocess(lr_frame)
            sr_tensor = model(lr_tensor)

            sr_img = postprocess(sr_tensor)

            # 生成对比图像（假设底部细节处理函数存在）
            lr_pil = to_image(lr_tensor.squeeze(0)).resize((lr_width * up_scale, lr_height * up_scale), Image.BICUBIC)
            sr_pil = to_image(sr_tensor.squeeze(0))

            lr_np = np.asarray(lr_pil)
            sr_np = np.asarray(sr_pil)
            top_image = np.concatenate([lr_np, sr_np], axis=1)

            # 假设process_crop返回底部细节图像（需根据实际逻辑实现）
            target_block_width = top_image.shape[1] // 10
            crop_lr = process_crop([lr_pil], target_block_width)
            crop_sr = process_crop([sr_pil], target_block_width)
            bottom_image = np.concatenate(crop_lr + crop_sr, axis=1)
            final_compare = np.concatenate([top_image, bottom_image], axis=0)

            # 保存临时帧
            sr_frame_path = sr_frame_folder / f"sr_frame_{k:04d}.png"
            compare_frame_path = compare_frame_folder / f"compare_frame_{k:04d}.png"

            cv2.imwrite(str(sr_frame_path), cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(str(compare_frame_path), cv2.cvtColor(final_compare, cv2.COLOR_BGR2RGB))

            test_bar.update(1)


    try:
        with VideoFileClip(input_lr_video_path) as original_video:
            original_audio = original_video.audio
            if original_audio.duration <= 0:
                raise ValueError("输入视频无有效音频轨道")

            sr_frame_list = [str(x) for x in sorted(
                sr_frame_folder.glob("*.png"),
                key=lambda x: int(x.stem.split("_")[-1])
            )]
            compare_frame_list = [str(x) for x in sorted(
                compare_frame_folder.glob("*.png"),
                key=lambda x: int(x.stem.split("_")[-1])
            )]

            sr_clip = ImageSequenceClip(sr_frame_list, fps=fps)
            compare_clip = ImageSequenceClip(compare_frame_list, fps=fps)

            # 尺寸校验（保持不变）
            assert sr_clip.size == (lr_width * up_scale, lr_height * up_scale), "高分辨率视频尺寸不一致"
            assert compare_clip.size == (lr_width * up_scale * 2 , lr_height * up_scale +sr_img.shape[1] // 5 ), \
                "对比视频尺寸不一致"

            sr_clip = sr_clip.set_audio(original_audio)
            compare_clip = compare_clip.set_audio(original_audio)

            sr_clip.write_videofile(
                str(output_sr_path + "\sr_demo.mp4"),
                codec="libx264",  # 视频编码
                audio_codec="aac",  # 音频编码
                fps=fps,  # 帧率
                verbose=True,  # 显示FFmpeg进度
                # 旧版参数替代方案（直接在参数中指定）
                audio_fps=44100,  # 音频采样率
                threads=4,  # 线程数（提升速度）
            )

            compare_clip.write_videofile(
                str(output_compare_path + "\compare_demo.mp4"),
                codec="libx264",  # 视频编码
                audio_codec="aac",  # 音频编码
                fps=fps,  # 帧率
                verbose=True,  # 显示FFmpeg进度
                # 旧版参数替代方案（直接在参数中指定）
                audio_fps=44100,  # 音频采样率
                threads=4,  # 线程数（提升速度）
            )
    finally:
        # 清理临时文件（保持不变）
        for frame in sr_frame_folder.glob("*.png"):
            frame.unlink(missing_ok=True)
        for frame in compare_frame_folder.glob("*.png"):
            frame.unlink(missing_ok=True)
        if not sr_frame_folder.iterdir():
            sr_frame_folder.rmdir()
        if not compare_frame_folder.iterdir():
            compare_frame_folder.rmdir()


def generate_bottom_contrast(lr_img, sr_img):
    """保持原底部细节对比逻辑（假设原函数存在）"""
    crop_size = sr_img.width // 5 - 9
    crop_lr = transforms.FiveCrop(crop_size)(lr_img)
    crop_sr = transforms.FiveCrop(crop_size)(sr_img)
    crop_lr = [np.asarray(transforms.Pad(10)(img)) for img in crop_lr]
    crop_sr = [np.asarray(transforms.Pad(10)(img)) for img in crop_sr]
    return np.concatenate([*crop_lr, *crop_sr], axis=1)



parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default="./video_inputs", type=str)
parser.add_argument('--output_path', default="./video_outputs", type=str)
parser.add_argument('--model_path', default="./pretrained_model", type=str)
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2],help='super resolution upscale factor')


if __name__ == "__main__":

    opt            =  parser.parse_args()
    video_path     = opt.video_path
    output_path    = opt.output_path
    upscale_factor =  opt.upscale_factor
    model_path     =  opt.model_path

    generate_sr_compare_video(video_path,output_path,output_path,model_path)







    #generate_lr_video(VIDEO_PATH,output_path,"lr_output.mp4",2,9,14)