# --------------------------------------------------------------
# BASE IMAGE: Sử dụng bản build mới nhất từ NVIDIA
# Đã bao gồm: PyTorch 2.x, CUDA 12.x, cuDNN tối ưu cho RTX 50 series
# --------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Thiết lập thông tin
LABEL maintainer="PromptMeo-Docker"

# Tắt các thông báo tương tác khi cài apt
ENV DEBIAN_FRONTEND=noninteractive

# Thiết lập thư mục làm việc
WORKDIR /workspace/PromptMeo

# --------------------------------------------------------------
# BƯỚC 1: Cài đặt các công cụ hệ thống cần thiết
# dos2unix: Để sửa lỗi xuống dòng của Windows
# git: Để cài đặt clip hoặc các thư viện từ github
# --------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    dos2unix \
    git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------
# BƯỚC 2: Cài đặt thư viện Python (Tận dụng Docker Cache)
# Copy file requirements trước để nếu code đổi thì không phải cài lại thư viện
# --------------------------------------------------------------
COPY requirements.txt .

# QUAN TRỌNG:
# 1. Xóa dòng chứa 'torch', 'torchvision' trong requirements.txt 
#    để dùng bản xịn có sẵn trong Docker (hỗ trợ RTX 5090).
# 2. Cài các thư viện còn lại.
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------
# BƯỚC 3: Copy toàn bộ mã nguồn vào
# --------------------------------------------------------------
COPY . .

# --------------------------------------------------------------
# BƯỚC 4: Cài đặt thư viện Dassl (Cốt lõi của PromptMeo)
# PromptMeo cần Dassl được cài đặt dưới dạng package
# --------------------------------------------------------------
RUN if [ -d "dassl" ]; then \
        cd dassl && \
        # Fix lỗi setup nếu có \
        rm -rf build dist && \
        python setup.py develop; \
    else \
        echo "WARNING: Dassl folder not found. Make sure you cloned submodules!"; \
    fi

# --------------------------------------------------------------
# BƯỚC 5: Xử lý lỗi Windows (CRLF) cho script
# Tìm tất cả file .sh và chạy dos2unix
# --------------------------------------------------------------
RUN find . -name "*.sh" -exec dos2unix {} \; && \
    chmod +x scripts/PromptMeo/*.sh

# Mặc định khi chạy container sẽ vào shell bash
CMD ["/bin/bash"]