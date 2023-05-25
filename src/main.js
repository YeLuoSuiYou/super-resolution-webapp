import { createApp } from 'vue'
import App from './App.vue'
import 'element-plus/dist/index.css'
import ElementPlus from 'element-plus'
import vue3PreviewImage from 'vue3-preview-image'


const app = createApp(App)
app.use(vue3PreviewImage)
app.use(ElementPlus)
app.mount('#app')

