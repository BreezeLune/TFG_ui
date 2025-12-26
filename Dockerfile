# TalkingGaussian UI System - Main Dockerfile
# 支持：训练、推理、实时对话、评测
# 基础镜像：CUDA 11.8（兼容 TalkingGaussian 的 CUDA 11.3 和 CosyVoice 的 CUDA 12.1）
# 注意：必须使用 devel 版本，因为需要编译 CUDA 扩展模块（diff-gaussian-rasterization, simple-knn, gridencoder）

FROM m.daocloud.io/docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 安装系统依赖
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gcc \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# 设置工作目录
WORKDIR /app

# 复制项目文件（先复制依赖文件，利用 Docker 缓存）
COPY requirements.txt /app/
COPY TalkingGaussian/environment.yml /app/TalkingGaussian/
COPY CosyVoice/requirements.txt /app/CosyVoice/
COPY TalkingGaussian/evaluation/requirements_eval.txt /app/TalkingGaussian/evaluation/

# 复制子模块目录（environment.yml 需要这些路径）
COPY TalkingGaussian/submodules /app/TalkingGaussian/submodules
COPY TalkingGaussian/gridencoder /app/TalkingGaussian/gridencoder

# 创建 TalkingGaussian 环境（Python 3.7.13, PyTorch 1.12.1, CUDA 11.3）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set remote_connect_timeout_secs 120.0 && \
    conda config --set remote_read_timeout_secs 600.0 && \
    conda config --set remote_max_retries 10 && \
    conda env create -f TalkingGaussian/environment.yml || \
    (echo "First attempt failed, retrying..." && sleep 10 && conda env create -f TalkingGaussian/environment.yml) && \
    conda clean -afy

# 创建 CosyVoice 环境（Python 3.10, PyTorch 2.3.1）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n cosyvoice python=3.10 -y && \
    conda activate cosyvoice && \
    pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r CosyVoice/requirements.txt && \
    conda deactivate && \
    conda clean -afy

# 创建评测环境（Python 3.10，用于 NIQE 等需要高版本 scikit-image 的指标）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n tg_eval python=3.10 -y && \
    conda activate tg_eval && \
    pip install --no-cache-dir torch==2.3.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir opencv-python scikit-image numpy scipy && \
    pip install --no-cache-dir -r TalkingGaussian/evaluation/requirements_eval.txt && \
    conda deactivate && \
    conda clean -afy

# 创建 NIQE 专用环境（Python 3.10，scikit-image 最新版）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n tg_niqe python=3.10 -y && \
    conda activate tg_niqe && \
    pip install --no-cache-dir -U scikit-image opencv-python numpy piq && \
    conda deactivate && \
    conda clean -afy

# 复制整个项目（排除 .dockerignore 中的文件）
COPY . /app/

# 安装 Flask 应用依赖（在 base 环境）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# 编译 TalkingGaussian 的扩展模块（diff-gaussian-rasterization, simple-knn, gridencoder）
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate talking_gaussian && \
    cd TalkingGaussian && \
    if [ -d "submodules/diff-gaussian-rasterization" ]; then \
        pip install ./submodules/diff-gaussian-rasterization; \
    fi && \
    if [ -d "submodules/simple-knn" ]; then \
        pip install ./submodules/simple-knn; \
    fi && \
    if [ -d "gridencoder" ]; then \
        pip install ./gridencoder; \
    fi && \
    conda deactivate && \
    conda clean -afy

# 设置权限
RUN chmod +x TalkingGaussian/run_talkinggaussian.sh && \
    chmod +x TalkingGaussian/scripts/train_xx.sh && \
    chmod +x CosyVoice/run_cosyvoice.sh && \
    chmod +x TalkingGaussian/evaluation/*.sh 2>/dev/null || true

# 创建必要的目录
RUN mkdir -p static/uploads/videos \
    static/uploads/audios \
    static/audios \
    static/videos \
    static/text \
    TalkingGaussian/output \
    TalkingGaussian/test_result \
    CosyVoice/test_result

# 暴露端口
EXPOSE 5001

# 默认命令：启动 Flask 应用
CMD ["python", "app.py"]

