# 面部识别服务 (Facial Recognition Service)

基于 FastAPI 和 PyTorch 的面部识别身份验证系统，支持面部检测、特征提取和身份验证。

## 功能特性

- 🔍 **面部检测**: 使用 MTCNN 进行精确的面部检测
- 🧠 **特征提取**: 基于 InceptionResnetV1 (VGGFace2) 预训练模型
- ⚡ **快速识别**: 通过余弦相似度进行身份验证
- 💾 **灵活存储**: Redis 临时存储 + 文件系统永久存储
- 🐳 **容器化部署**: 完整的 Docker 支持
- 🔄 **异步处理**: FastAPI 异步请求处理
- 🏥 **健康检查**: 内置服务健康监控

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   客户端请求    │───▶│   FastAPI 服务   │───▶│   面部识别引擎   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Redis 缓存     │    │  嵌入向量存储    │
                       └─────────────────┘    └─────────────────┘
```

## 快速开始

### 前置要求

- Docker & Docker Compose
- 预训练模型文件: `20180402-114759-vggface2.pt`

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd facial-recognition-py
```

### 2. 准备模型文件

下载预训练模型文件并放置到 `models/` 目录：

```bash
mkdir -p models
# 将 20180402-114759-vggface2.pt 文件放到 models/ 目录下
```

### 3. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
MODEL_PATH=/app/models/20180402-114759-vggface2.pt
EMBEDDING_THRESHOLD=0.8
```

### 4. 启动服务

```bash
# 一键启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f face-service
```

### 5. 验证部署

- 🌐 服务地址: http://localhost:8000
- 📚 API 文档: http://localhost:8000/docs
- 🔧 Redis: localhost:6379

## API 使用说明

### 面部识别接口

```bash
POST /recognize
Content-Type: multipart/form-data
```

**参数:**
- `image`: 图片文件
- `employee_id`: 员工ID（可选，用于首次注册）

**示例:**

```bash
# 首次注册
curl -X POST "http://localhost:8000/recognize" \
  -H "accept: application/json" \
  -F "image=@/path/to/photo.jpg" \
  -F "employee_id=emp_001"

# 身份验证
curl -X POST "http://localhost:8000/recognize" \
  -H "accept: application/json" \
  -F "image=@/path/to/photo.jpg"
```

**响应示例:**

```json
{
  "success": true,
  "message": "Face registered successfully",
  "employee_id": "emp_001",
  "confidence": 0.95
}
```

## 云服务器部署

### 1. 服务器环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo apt install docker-compose-plugin -y

# 添加用户到 docker 组
sudo usermod -aG docker $USER
```

### 2. 部署步骤

```bash
# 克隆仓库
git clone <your-repo-url>
cd facial-recognition-py

# 上传模型文件到 models/ 目录
# 可使用 scp 或其他方式上传

# 配置生产环境变量
cp .env.example .env
nano .env  # 根据需要修改配置

# 启动服务
docker-compose up -d

# 配置防火墙（如果需要）
sudo ufw allow 8000
sudo ufw allow 6379  # 仅在需要外部访问 Redis 时
```

### 3. 生产环境配置

编辑 `docker-compose.yml` 进行生产优化：

```yaml
# 移除端口暴露（使用反向代理）
# ports:
#   - "8000:8000"

# 添加资源限制
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'

# 配置日志
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 4. 反向代理配置 (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 支持大文件上传
        client_max_body_size 10M;
    }
}
```

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `REDIS_HOST` | redis | Redis 主机地址 |
| `REDIS_PORT` | 6379 | Redis 端口 |
| `REDIS_DB` | 0 | Redis 数据库编号 |
| `MODEL_PATH` | /app/models/20180402-114759-vggface2.pt | 模型文件路径 |
| `EMBEDDING_THRESHOLD` | 0.8 | 识别阈值 |

### 存储策略

- **Redis**: 临时嵌入向量存储，7天TTL，格式：`guard:{employeeId}`
- **文件系统**: 永久嵌入向量存储，路径：`tests/embeddings/{employeeId}.npy`

### 识别阈值

推荐阈值设置：
- **高安全性**: 0.9+ (误识率低，拒识率高)
- **平衡模式**: 0.8 (推荐)
- **高通过率**: 0.6-0.7 (误识率相对较高)

## 性能优化

### 1. GPU 加速

如果服务器有GPU，修改 `requirements.txt`：

```txt
torch>=2.0.0+cu118  # 根据 CUDA 版本选择
torchvision>=0.15.0+cu118
```

### 2. 模型预热

服务启动时自动进行模型预热，提高首次请求响应速度。

### 3. 并发处理

通过 `uvicorn` 配置调整并发：

```bash
uvicorn face_service:app --host 0.0.0.0 --port 8000 --workers 4
```

## 监控与日志

### 查看日志

```bash
# 查看服务日志
docker-compose logs -f face-service

# 查看 Redis 日志
docker-compose logs -f redis
```

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/docs

# 检查 Redis 连接
docker-compose exec redis redis-cli ping
```

## 故障排除

### 常见问题

1. **模型文件未找到**
   ```
   确保 models/20180402-114759-vggface2.pt 文件存在
   ```

2. **Redis 连接失败**
   ```bash
   docker-compose exec redis redis-cli ping
   ```

3. **内存不足**
   ```
   增加服务器内存或调整 Docker 内存限制
   ```

4. **端口占用**
   ```bash
   sudo lsof -i :8000
   sudo lsof -i :6379
   ```

## 安全注意事项

- 🔒 在生产环境中使用 HTTPS
- 🛡️ 设置适当的防火墙规则
- 🔐 保护 Redis 实例（密码认证）
- 📊 定期备份嵌入向量数据
- 🔍 监控异常识别行为

## 开发指南

### 本地开发

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行服务
uvicorn face_service:app --reload
```

### 测试

```bash
# 运行测试
python tests/test1.py
python tests/redis_test.py
python tests/save_embed.py
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 Issue 或联系维护者。