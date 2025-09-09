from facenet_pytorch import MTCNN, InceptionResnetV1
import torch, numpy as np
from PIL import Image, ImageOps

import redis
r = redis.Redis(host='127.0.0.1', port=6379, db=0)

# 1) 初始化检测器和模型
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(classify=False).eval()

state_dict = torch.load("../models/20180402-114759-vggface2.pt", map_location='cpu')
resnet.load_state_dict(state_dict, strict=False)  

resnet.eval()

# 2) 封装一个函数：从图片路径输出 L2 归一化后的 embedding
def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"No face detected in {image_path}")
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).squeeze().numpy()
    emb /= np.linalg.norm(emb)
    return emb

# 3) 指定两张图片
image1 = "images/me4.jpg"
# image2 = "images/me3.jpg"

# 4) 生成 embedding
emb_registered = get_embedding(image1)
emb_checkin = r.get('guard:test:emb')
emb_checkin = np.frombuffer(emb_checkin, dtype=np.float32)

# 5) 计算相似度
similarity = float(np.dot(emb_registered, emb_checkin))
print(f"Cosine similarity: {similarity:.4f}")

# 6) 阈值判断
THRESHOLD = 0.8
if similarity >= THRESHOLD:
    print("✅ 匹配成功")
else:
    print("❌ 匹配失败")