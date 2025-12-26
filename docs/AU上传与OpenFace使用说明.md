## AU 文件上传接口

- 路由：`POST /upload_au`
- 表单字段：
  - `project_id`（必填）：对应数据目录名，如 `May`，将保存到 `TalkingGaussian/data/<project_id>/au.csv`
  - `au_file`（必填）：`au.csv` 文件，格式仅允许 `.csv`
- 服务器保存路径：`TalkingGaussian/data/<project_id>/au.csv`
- 适用场景：
  - 已在本地/其他环境用 OpenFace 生成了 AU（`au.csv`），直接上传到服务器
  - 前端无需改训练/推理流程，放置到数据目录后即可使用

## OpenFace 使用建议

1) 推荐方式（不嵌套 Docker）
   - 在本地或工具机直接安装/运行 OpenFace，生成 `au.csv`（以及可选的 landmarks 等）。
   - 生成后：
     - 方式 A：用 `/upload_au` 接口上传 `au.csv`
     - 方式 B：直接拷贝到服务器 `TalkingGaussian/data/<ID>/au.csv`
   - 优点：避免 docker-in-docker，简单稳定。

2) 如果要在项目中保留 OpenFace 资源
   - 可在仓库中放置压缩包位置（建议）：
     - `TalkingGaussian/third_party/openface/` 或 `TalkingGaussian/tools/openface/`
     - 压缩包命名示例：`openface_win.zip`、`openface_linux.tar.gz`
   - 仅存放压缩包与使用说明，不建议在最终镜像里再调用 docker 版 OpenFace。

3) 不需要放到 `static/`
   - AU 是训练/推理用的内部数据，应放在 `TalkingGaussian/data/<ID>/au.csv`。
   - `static/` 仅用于前端可直接访问的静态资源（视频/音频/文本）。

## 前端处理建议

- 目前前端未提供“上传 AU”按钮，若需要可新增一个简单表单：
  - 字段：`project_id`（文本）、`au_file`（文件）
  - 提交到 `POST /upload_au`
- 如果不改前端，也可用 Postman/curl 直接调接口上传 AU。训练/推理页面保持原有视频/模型路径填写即可。

## 使用流程示例

1. 本地跑 OpenFace 得到 `au.csv`
2. 将 `au.csv` 上传/拷贝到服务器：
   - 方案 A：`curl -F "project_id=May" -F "au_file=@au.csv" http://<host>:5001/upload_au`
   - 方案 B：直接放到 `TalkingGaussian/data/May/au.csv`
3. 训练/推理时，前端照常填写：
   - 训练视频：`static/uploads/videos/May.mp4`（系统会复制到 `TalkingGaussian/data/May/May.mp4`）
   - 模型路径：`output/May`
   - 参考音频：`static/uploads/audios/may_reference.wav`（训练后自动复制）或其他

