# 图片超分辨率Web应用

本项目的前端由Vue3开发，后端由Flask开发，通过HTTP请求通信。后端调用SRGAN网络实现图像超分辨率

## 推荐的IDE设置

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur) + [TypeScript Vue Plugin (Volar)](https://marketplace.visualstudio.com/items?itemName=Vue.vscode-typescript-vue-plugin).

## 运行项目

```shell
git clone https://gitee.com/the_night_falls_with_you/Super-resolution.git
cd Super-resolution
```

### 进入conda环境，安装相关的python依赖

```shell
conda activate my_env
pip install -r requirements.txt
```

**warning：本项目强烈建议装好cuda版本的pytorch后再进行clone，cpu版本的跑不动本项目的模型推理部分，执意使用cpu推理，请更改代码**

```python
# ./back/server.py line49
#super_res_image = generate_super_resolution_image(image, lower_res_filename_save, super_res_filename_save, device_type="cuda")
super_res_image = generate_super_resolution_image(image, lower_res_filename_save, super_res_filename_save, device_type="cpu")
```

### 安装Vue相关依赖

```shell
npm install
```

### 启动前后端

```shell
# 启动前端
npm run dev
# 切换一个powershell，启动后端
cd back
python server.py
```

即可打开命令行中的前端地址访问本Web应用
