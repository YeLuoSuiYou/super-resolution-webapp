from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import os

app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    # 获取图片文件
    image = request.files.get("image")
    if not image:
        return jsonify({"error": 1001, "msg": "未上传图片"})
    
    # 生成随机文件名
    ext = os.path.splitext(image.filename)[1]
    filename = str(uuid.uuid1()) + ext

    # 保存图片文件
    image.save(os.path.join("image/", filename))

    # 返回图片URL
    return jsonify({"url": request.host + "/image/" + filename,
                    "filename": filename})


@app.route("/upload", methods=["OPTIONS"])
def upload_options():
    return "", 200


@app.route("/image/<filename>")
def image(filename):
    return send_from_directory("image/", filename)


if __name__ == "__main__":
    app.run(debug=True)
