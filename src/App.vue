<template>
    <div id="app">
        <div v-if="!loggedIn">
            <Login @login-success="handleLoginSuccess"/>
        </div>
        <div v-else>
            <Upload :fileList="fileList" />
            <ImageContainer v-for="(file, index) in fileList" :key="index" :file="file" @remove="handleRemove" />
            <div class="center-bottom">Jason_Ye    2023</div>
        </div>
    </div>
</template>

<script>
import Upload from './components/Upload.vue';
import ImageContainer from './components/ImageContainer.vue';
import Login from './components/Login.vue';

export default {
    components: {
        Upload,
        ImageContainer,
        Login,
    },
    data() {
        return {
            fileList: [],
            loggedIn: false,
        };
    },
    methods: {
        handleRemove(file) {
            const index = this.fileList.indexOf(file);
            if (index !== -1) {
                this.fileList.splice(index, 1);
            }
        },
        handleLoginSuccess() {
            this.loggedIn = true;
        },
    },
};
</script>

<style scoped>
.center-bottom {
  position: absolute;
  bottom: 0;
  left: 50%;
  color: grey;
  transform: translateX(-50%);
}
</style>