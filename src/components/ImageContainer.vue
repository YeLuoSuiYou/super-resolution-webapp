<template>
    <div class="image-container">
        <img :src="file.url" :alt="file.name" class="uploaded-image preview-image" @click="$preview(file.url)" />
        <div class="button-container">
            <button class="download-button" @click="downloadImage(file.url, file.name)" style="z-index: 1;">下载图片</button>
        </div>
        <div class="button-container">
            <button class="delete-button" @click="handleRemove(file)" style="z-index: 1;">删除图片</button>
        </div>
    </div>
</template>

<script>
export default {
    name: 'ImageContainer',
    props: {
        file: {
            type: Object,
            required: true,
        },
    },
    methods: {
        downloadImage(url, name) {
            fetch(url)
                .then(response => response.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = name;
                    link.click();
                    URL.revokeObjectURL(url);
                });
        },
        handleRemove(file) {
            this.$emit('remove', file);
        },
    },
};
</script>

<style scoped>
/* CSS for ImageContainer component */
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