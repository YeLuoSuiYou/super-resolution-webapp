from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    # 获取图片文件
    image = request.files.get("image")
    if not image:
        return jsonify({"error": 1001, "msg": "未上传图片"})

    # 保存图片文件
    filename = image.filename or "default.png"
    image.save(os.path.join("image/", filename))

    # 返回图片URL
    filename = image.filename or "default.png"
    print(request.host + "/image/" + filename)
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
