from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference import generate_super_resolution_image
import uuid
import os
import glob
import copy

app = Flask(__name__)
CORS(app)

def limit_number_images(max_images: int) -> None:
    # Remove excess images from the "image/lr/" directory
    lr_images = glob.glob("image/lr/*")
    if len(lr_images)+1 > max_images:
        for i in range(len(lr_images)+1 - max_images):
            os.remove(lr_images[i])

    # Remove excess images from the "image/sr/" directory
    sr_images = glob.glob("image/sr/*")
    if len(sr_images)+1 > max_images:
        for i in range(len(sr_images)+1 - max_images):
            os.remove(sr_images[i])


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    # 获取图片文件
    image = request.files.get("image")

    if not image:
        return jsonify({"error": 1001, "msg": "未上传图片"})

    # 限制图片数量
    limit_number_images(5)

    # 生成随机文件名
    ext = os.path.splitext(image.filename)[1] # type: ignore
    filename = str(uuid.uuid1()) + ext


    # 保存图片文件地址
    lower_res_filename = "lr_" + filename
    lower_res_filename_save = os.path.join("image/lr/", lower_res_filename)
    super_res_filename = "sr_" + filename
    super_res_filename_save = os.path.join("image/sr/", super_res_filename)

    # 调用超分辨率模型
    super_res_image = generate_super_resolution_image(image, lower_res_filename_save, super_res_filename_save, device_type="cuda")

    # 返回超分辨率图片URL
    return jsonify({"url": request.host + "/image/sr/" + super_res_filename,
                    "filename": super_res_filename})

@app.route("/upload", methods=["OPTIONS"])
def upload_options():
    return "", 200


@app.route("/image/sr/<filename>")
def sr_image(filename):
    return send_from_directory("image/sr/", filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, )

