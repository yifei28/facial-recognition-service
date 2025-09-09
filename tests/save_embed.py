from facenet_pytorch import MTCNN, InceptionResnetV1
import torch, numpy as np
from PIL import Image, ImageOps

# 初始化
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(classify=False).eval()

state_dict = torch.load("../models/20180402-114759-vggface2.pt", map_location='cpu')
resnet.load_state_dict(state_dict, strict=False)  

resnet.eval()

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    face = mtcnn(img)
    if face is None:
        raise ValueError("未检测到人脸")
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).squeeze().numpy()
    # 归一化
    emb /= np.linalg.norm(emb)
    return emb

# 1) 生成注册照 embedding
emb_reg = get_embedding("images/me4.jpg")
# 存到磁盘
np.save("embeddings/guard1.npy", emb_reg)

print("✅ 已保存 embeddings/guard_test.npy")