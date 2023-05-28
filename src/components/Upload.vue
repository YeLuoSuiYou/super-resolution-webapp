<template>
    <h1>图片超分辨率Web应用</h1>
    <el-upload class="upload-demo" drag :name="'image'" :action="'http://127.0.0.1:5000/upload'" multiple
        :before-upload="beforeUpload" :on-remove="handleRemove" :on-success="handleSuccess" :before-remove="beforeRemove"
        :show-file-list="false">
        <i class="el-icon-upload"></i>
        <div class="el-upload__text">将文件拖到此处,或<em>点击上传</em></div>
        <div class="el-upload__tip" slot="tip">只能上传jpg/png文件,且不超过500kb</div>
    </el-upload>
</template>

<script>
export default {
    name: 'Upload',
    props: {
        fileList: {
            type: Array,
            default: () => [],
        },
    },
    methods: {
        beforeUpload(file) {
            const isJPG = file.type === "image/jpeg";
            const isPNG = file.type === "image/png";
            const isLt2M = file.size / 1024 / 1024 < 30;
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
    },
};
</script>

<style scoped>
h1 {
    background-image: url(/background/school.jpg);
    background-size: cover;
    background-position: center;
    color: white;
    text-align: center;
    font-size: 50px;
    padding: 50px 0;
}
</style>