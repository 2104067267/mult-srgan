from flask import Flask, request, send_file, jsonify, make_response
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import NewGenerator
from utils import load_model, evaluate_super_resolution, display_image
from flask_cors import CORS
import logging
from pathlib import Path
import base64
import zipfile
import matplotlib.pyplot as plt
from tqdm import tqdm

# 初始化 Flask 应用并启用跨域支持
app = Flask(__name__)
CORS(app)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = "pretrained_model"

try:
    # 加载预训练的模型
    model = NewGenerator(2)
    load_model(model, model_path + "/generator", 'generator')
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor()
])


@app.route('/api/image-super-resolution', methods=['POST'])
def image_super_resolution():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        img = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            output_tensor = model(input_tensor)

        display_image([input_tensor.squeeze(), output_tensor.squeeze()])

        # 将输出张量转换为图像
        output_img = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # 将图像数据转换为 base64 编码
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        return jsonify({
            "image": img_base64,
        })
    except Exception as e:
        logger.error(f"Error during image super-resolution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/two-folders-super-resolution', methods=['POST'])
def two_folders_super_resolution():
    try:
        low_res_files = request.files.getlist('lowResFiles')
        high_res_files = request.files.getlist('highResFiles')

        if not low_res_files or not high_res_files:
            return jsonify({"error": "No files provided"}), 400

        reconstructed_images = []
        psnr_list = []
        ssim_list = []
        entropy_list = []

        # 创建保存重建图像的文件夹
        output_folder = Path("reconstructed_folder")
        output_folder.mkdir()
        print("create folder successfully")

        for index, (low_res_file, high_res_file) in enumerate(tqdm(zip(low_res_files, high_res_files))):
            low_res_img = Image.open(low_res_file.stream).convert('RGB')
            high_res_img = Image.open(high_res_file.stream).convert('RGB')

            low_res_tensor = preprocess(low_res_img).unsqueeze(0).to(device)
            high_res_tensor = preprocess(high_res_img).unsqueeze(0).to(device)

            # 模型推理
            with torch.no_grad():
                output_tensor = model(low_res_tensor)

            # display_image([low_res_tensor.squeeze(), output_tensor.squeeze()])

            psnr, ssim, entropy = evaluate_super_resolution(
                high_res_tensor.squeeze().permute(1, 2, 0).cpu().numpy(),
                output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            )

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            entropy_list.append(entropy)

            # 将输出张量转换为图像
            output_img = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
            img_byte_arr = io.BytesIO()
            output_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # 将图像数据转换为 base64 编码
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            reconstructed_images.append(img_base64)

            # 保存重建后的图像到文件夹
            output_path = output_folder / f"{index}.png"
            output_img.save(output_path)

        # 为每个指标生成单独的折线图
        def generate_plot(data, metric_name, output_folder):
            plt.figure(figsize=(10, 6))
            plt.plot(data, label=metric_name)
            plt.xlabel('Image Index')
            plt.ylabel('Metric Value')
            plt.title(f'{metric_name} for Reconstructed Images')

            # 计算平均值
            average = sum(data) / len(data)

            # 绘制代表平均值的水平直线
            plt.axhline(y=average, color='r', linestyle='--', label=f'Average {metric_name}')

            # 添加文本注释突出显示平均值
            plt.text(0.5, 0.9, f'Average {metric_name}: {average:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes,
                     color='r',
                     fontsize=14,
                     bbox=dict(facecolor='white', edgecolor='r', boxstyle='round'))

            plt.legend()
            plot_byte_arr = io.BytesIO()
            plt.savefig(plot_byte_arr, format='PNG')
            plot_byte_arr.seek(0)
            plot_base64 = base64.b64encode(plot_byte_arr.getvalue()).decode('utf-8')

            # 保存指标图像到文件夹
            plot_path = output_folder / f"{metric_name}_plot.png"
            plt.savefig(plot_path)

            plt.close()
            return plot_base64

        psnr_plot = generate_plot(psnr_list, 'PSNR', output_folder)
        ssim_plot = generate_plot(ssim_list, 'SSIM', output_folder)
        entropy_plot = generate_plot(entropy_list, 'Entropy', output_folder)

        return jsonify({
            "reconstructedImages": reconstructed_images,
            "psnr_plot": psnr_plot,
            "ssim_plot": ssim_plot,
            "entropy_plot": entropy_plot
        })
    except Exception as e:
        logger.error(f"Error during two-folders super-resolution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download-reconstructed-folder', methods=['GET'])
def download_reconstructed_folder():
    try:
        # 这里假设你有一个文件夹存储重建后的图像
        folder_path = Path("reconstructed_folder")
        if not folder_path.exists():
            return jsonify({"error": "Reconstructed folder not found"}), 404

        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for file in folder_path.iterdir():
                zf.write(file, file.name)
        memory_file.seek(0)

        response = make_response(memory_file.getvalue())
        response.headers.set('Content-Type', 'application/zip')
        response.headers.set('Content-Disposition', 'attachment', filename='reconstructed_folder.zip')
        return response
    except Exception as e:
        logger.error(f"Error during downloading reconstructed folder: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/video-super-resolution', methods=['POST'])
def video_super_resolution():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # 视频超分辨率处理相对复杂，这里简单返回原视频
        file_content = file.read()
        return send_file(
            io.BytesIO(file_content),
            mimetype=file.mimetype
        )
    except Exception as e:
        logger.error(f"Error during video super-resolution: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
