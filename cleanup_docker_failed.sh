#!/bin/bash
# Docker 构建失败后的清理脚本

echo "=========================================="
echo "清理 Docker 构建失败的残留文件"
echo "=========================================="

# 1. 停止所有构建进程
echo "[1/5] 停止构建进程..."
pkill -f "docker build" 2>/dev/null
screen -S docker_build -X quit 2>/dev/null
sleep 2

# 2. 清理 Docker 系统
echo "[2/5] 清理 Docker 系统..."
docker system prune -af 2>/dev/null
docker builder prune -af 2>/dev/null

# 3. 清理未使用的镜像层
echo "[3/5] 清理未使用的镜像..."
docker image prune -af 2>/dev/null

# 4. 清理 Docker 临时文件
echo "[4/5] 清理临时文件..."
find /var/lib/docker -name "*.tmp" -type f -delete 2>/dev/null
rm -rf /var/lib/docker/tmp/* 2>/dev/null

# 5. 显示清理结果
echo "[5/5] 清理完成！"
echo ""
df -h / | tail -1
echo ""
docker system df

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="

