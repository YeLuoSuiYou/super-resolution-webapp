<template>
    <div id="app">
        <el-upload class="upload-demo" drag :name="'image'" :action="'http://127.0.0.1:5000/upload'" multiple
            :before-upload="beforeUpload" :on-remove="handleRemove" :on-success="handleSuccess"
            :before-remove="beforeRemove" :show-file-list="false">
            <i class="el-icon-upload"></i>
            <div class="el-upload__text">将文件拖到此处,或<em>点击上传</em></div>
            <div class="el-upload__tip" slot="tip">只能上传jpg/png文件,且不超过500kb</div>
        </el-upload>
        <div v-for="(file, index) in fileList" :key="index" class="image-container">
            <img :src="file.url" :alt="file.name" class="uploaded-image preview-image" @click="$preview(file.url)" />
            <div class="button-container">
                <button class="download-button" @click="downloadImage(file.url, file.name)"
                    style="z-index: 1;">下载图片</button>
            </div>
            <div class="button-container">
                <button class="delete-button" @click="handleRemove(file)" style="z-index: 1;">删除图片</button>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            fileList: [],
        };
    },
    methods: {
        beforeUpload(file) {
            const isJPG = file.type === "image/jpeg";
            const isPNG = file.type === "image/png";
            const isLt2M = file.size / 1024 / 1024 < 100;
            if (!isJPG && !isPNG) {
                this.$message.error("上传图片只能是 JPG/PNG 格式!");
            }
            if (!isLt2M) {
                this.$message.error("上传图片大小不能超过 30MB!");
            }
            return (isJPG || isPNG) && isLt2M;
        },
        handleSuccess(response) {
            this.fileList.push({
                url: "http://" + response.url,
                name: response.filename,
            });
            this.$nextTick(() => {
                this.$preview('http://' + response.url);
                console.log(response.url);
            });
        },
        handleRemove(file) {
            const index = this.fileList.indexOf(file);
            if (index !== -1) {
                this.fileList.splice(index, 1);
            }
        },
        beforeRemove(file) {
            this.fileList.splice(this.fileList.indexOf(file), 1);
        },
        async downloadImage(imgUrl, filename) {
            try {
                const response = await fetch(imgUrl);
                const data = await response.blob();
                const url = URL.createObjectURL(data);
                const link = document.createElement("a");
                link.href = url;
                link.download = filename;
                link.click();
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error("下载图片失败:", error);
            }
        },
    },
};
</script>

<style scoped>
.image-container {
    display: inline-block;
    position: relative;
    margin: 10px;
    width: 400px;
    height: 400px;
    text-align: center;
}

.uploaded-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.button-container {
    position: relative;
    width: 100%;
    height: 100%;
}

.download-button {
    position: absolute;
    bottom: 90%;
    left: 25%;
    transform: translateX(-50%);
    background-color: rgb(130, 176, 210);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 10px;
    cursor: pointer;
}

.download-button:hover {
    background-color: rgb(142, 207, 201);
}

.delete-button {
    position: absolute;
    bottom: 190%;
    right: 30%;
    transform: translateX(50%);
    background-color: rgb(130, 176, 210);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 10px;
    cursor: pointer;
}

.delete-button:hover {
    background-color: rgb(142, 207, 201);
}

.preview-image {
    filter: brightness(100%);
}

.preview-image:hover {
    filter: brightness(70%);
    cursor: pointer;
}
</style>