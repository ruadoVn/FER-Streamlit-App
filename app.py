import streamlit as st
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import pandas as pd

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG DAN (Deep Alignment Network)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256))
        self.conv_3x3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.conv_1x3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)), nn.BatchNorm2d(512))
        self.conv_3x1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)), nn.BatchNorm2d(512))
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True) 
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 512), nn.Sigmoid()    
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        return sa * y

class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)
        return ca

class DAN(nn.Module):
    # Đã sửa lại để tương thích PyTorch mới (bỏ parameter pretrained cũ)
    def __init__(self, num_class=7, num_head=4):
        super(DAN, self).__init__()
        resnet = models.resnet18(weights=None) # Cập nhật chuẩn mới của Torchvision
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, f"cat_head{i}", CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, f"cat_head{i}")(x))
        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)
        out = self.bn(self.fc(heads.sum(dim=1)))
        return out, x, heads


# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ CACHE (TỐI ƯU RAM SERVER)
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MTCNN (Tìm khuôn mặt)
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Load DAN Model
    model = DAN(num_class=7)
    # LƯU Ý: Đảm bảo bạn đã upload file này lên cùng thư mục code
    checkpoint = torch.load('rafdb_model.pth', map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return mtcnn, model, device

mtcnn, model, device = load_models()

labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. GIAO DIỆN STREAMLIT VÀ XỬ LÝ LOGIC
# ==========================================
st.set_page_config(page_title="AI Emotion Detection", page_icon="🎭", layout="wide")
st.title("🎭 Hệ Thống Nhận Diện Cảm Xúc (Deep Alignment Network)")
st.markdown("Được hỗ trợ bởi mạng **DAN** và công nghệ tìm kiếm khuôn mặt **MTCNN**.")

# Cột Menu
st.sidebar.header("Tùy chọn đầu vào")
input_mode = st.sidebar.radio("Chọn nguồn ảnh:", ("Tải ảnh lên (Upload)", "Chụp từ Webcam"))

def process_image(img_file):
    img_pil = Image.open(img_file).convert('RGB')
    
    with st.spinner('AI đang quét khuôn mặt và phân tích...'):
        boxes, _ = mtcnn.detect(img_pil)
        
        if boxes is None:
            st.error("❌ Không tìm thấy khuôn mặt nào trong ảnh!")
            return

        st.success(f"✅ Tìm thấy {len(boxes)} khuôn mặt!")
        
        # Tạo công cụ vẽ để vẽ khung lên ảnh
        draw = ImageDraw.Draw(img_pil)
        col1, col2 = st.columns([2, 1])
        chart_data = None
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1) # Chống lỗi tọa độ âm
            
            # Cắt mặt và đưa vào mô hình dự đoán
            face = img_pil.crop((x1, y1, x2, y2))
            face_tensor = transform(face).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out, _, _ = model(face_tensor)
                prob = torch.nn.functional.softmax(out, dim=1)
                pred_idx = torch.argmax(prob, dim=1).item()
                conf = prob[0][pred_idx].item()
                
                if idx == 0: # Chỉ lưu data vẽ biểu đồ cho khuôn mặt đầu tiên
                    chart_data = {labels[i]: float(prob[0][i]) * 100 for i in range(7)}
            
            # Vẽ Box và Text lên ảnh
            draw.rectangle(((x1, y1), (x2, y2)), outline="#00FF00", width=4)
            text = f"{labels[pred_idx]} ({conf*100:.1f}%)"
            
            # Mẹo nhỏ để vẽ viền đen cho chữ dễ đọc trên nền sáng
            draw.text((x1, max(0, y1 - 20)), text, fill="white", stroke_width=2, stroke_fill="black")

        # Hiển thị ảnh bên trái
        with col1:
            st.image(img_pil, caption="Kết quả Phân tích từ AI", use_container_width=True)
            
        # Hiển thị biểu đồ bên phải
        with col2:
            st.subheader("Phân tích Cảm xúc Chi tiết")
            if chart_data:
                df = pd.DataFrame({
                    "Cảm xúc": list(chart_data.keys()),
                    "Tỷ lệ (%)": list(chart_data.values())
                }).set_index("Cảm xúc")
                st.bar_chart(df, use_container_width=True)

# Lắng nghe sự kiện người dùng
if input_mode == "Tải ảnh lên (Upload)":
    uploaded_file = st.file_uploader("Vui lòng chọn một bức ảnh", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        process_image(uploaded_file)

elif input_mode == "Chụp từ Webcam":
    camera_photo = st.camera_input("Hãy mỉm cười hoặc biểu diễn một cảm xúc!")
    if camera_photo is not None:
        process_image(camera_photo)