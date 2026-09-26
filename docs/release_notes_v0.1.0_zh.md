# v0.1.0 发布说明（中文）

[English release notes](https://github.com/williamQ96/schema_study/releases/tag/v0.1.0) | **简体中文**

发布四类 paper–dataset 证据流程、可替换模型配置、Docker CUDA 镜像和 Mercury Apptainer 启动器。

- [使用指南](https://github.com/williamQ96/schema_study/blob/v0.1.0/docs/schema_study_user_guide_zh.md)：PDF 输入准备、独立 dataset 解析、corpus、运行/续跑及 packet 验证。
- [部署指南](https://github.com/williamQ96/schema_study/blob/v0.1.0/docs/schema_study_deployment_guide_zh.md)：Docker Hub → SIF、挂载、模型替换、checkpoint、冻结和 GPU 预检。
- [Docker Hub](https://hub.docker.com/r/plalelab/schema-study)：`plalelab/schema-study:0.1.0-cuda13`（Linux amd64）。

固定镜像内容：

```bash
docker pull plalelab/schema-study@sha256:c6abbb7f853e295dcce0634029eb991a1f1d488bb548c986b40998b1476db45e
apptainer pull schema-study-0.1.0.sif docker://docker.io/plalelab/schema-study@sha256:c6abbb7f853e295dcce0634029eb991a1f1d488bb548c986b40998b1476db45e
```

源码 commit：`07e5f989b127d25783d9c65a6b83c414b4562bf3`。OCI digest 与转换后的 SIF SHA-256 不同，实验应另行冻结实际 SIF 文件哈希。

验证：Linux 容器 113 项测试通过；Windows 112 项通过、1 项因符号链接权限跳过；[GitHub Actions](https://github.com/williamQ96/schema_study/actions/runs/36266428225) 通过。文档中的三个 Python 准备示例已执行验证。离线流程完成 11 次 mock 派发，续跑 0 次新派发，packet 完整性和派生验证通过，真实模型调用数为 0。

镜像包含代码和依赖，PDF、dataset、模型权重和凭据由外部挂载。原有历史冻结文件哈希未变。Mercury 现场 GPU/SIF qualification、真实推理与独立语义评价尚未执行；示例模型配置保持 draft。
