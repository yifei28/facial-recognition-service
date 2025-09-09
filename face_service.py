import os, io, torch, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageOps
from dotenv import load_dotenv
import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False, db=int(os.getenv("REDIS_DB", 0)))

load_dotenv()
app = FastAPI()

# 设备 & 模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(classify=False).to(DEVICE).eval()

# 加载权重
state_dict = torch.load(os.getenv("MODEL_PATH"), map_location=DEVICE)
resnet.load_state_dict(state_dict, strict=False)
resnet.eval()

# 阈值
THRESHOLD = float(os.getenv("EMBEDDING_THRESHOLD", 0.8))

# 加载 embeddings 目录下所有 .npy 文件
registered_embeddings = {}
emb_dir = "tests/embeddings"
for fname in os.listdir(emb_dir):
    if fname.endswith(".npy"):
        guard_id = fname[:-4]    # 去掉后缀当作 guardId
        path = os.path.join(emb_dir, fname)
        registered_embeddings[guard_id] = np.load(path)

class RecognizeResult(BaseModel):
    success: bool
    guardId: str = None
    similarity: float = None
    message: str = None

@app.post("/recognize", response_model=RecognizeResult)
async def recognize(faceImage: UploadFile = File(...), employeeId: str = Form(...)):
    data = await faceImage.read()
    return await run_in_threadpool(_recognize_sync, data, employeeId)

def _recognize_sync(data: bytes, employee_id: str) -> RecognizeResult:
    redis_key = f"guard:{employee_id}"
    
    # 如果 Redis 没这个 key：直接注册 + 成功返回
    if not rds.exists(redis_key):
        # 检测人脸
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        face = mtcnn(img)
        if face is None:
            raise HTTPException(400, "未检测到人脸")
        face = face.to(DEVICE)
        with torch.no_grad():
            emb = resnet(face.unsqueeze(0)).cpu().squeeze().numpy()
        emb /= np.linalg.norm(emb)

        # 存入 Redis
        rds.set(redis_key, emb.tobytes(), ex= 7 * 24 * 60 * 60) # 30天有效
        return RecognizeResult(
            success=True,
            guardId=employee_id,
            similarity=1.0,
            message="首次注册并识别成功"
        )

    # 否则走正常比对逻辑
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    face = mtcnn(img)
    if face is None:
        raise HTTPException(400, "未检测到人脸")
    face = face.to(DEVICE)
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).cpu().squeeze().numpy()
    emb /= np.linalg.norm(emb)

    # 加载注册向量
    ref_bytes = rds.get(redis_key)
    ref_emb = np.frombuffer(ref_bytes, dtype=np.float32)
    similarity = float(np.dot(emb, ref_emb))

    if similarity >= THRESHOLD:
        return RecognizeResult(success=True, guardId=employee_id, similarity=similarity, message="识别成功")
    return RecognizeResult(success=False, similarity=similarity, message="人脸不匹配")