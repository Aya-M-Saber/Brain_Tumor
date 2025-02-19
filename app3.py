import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل النموذج
class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load('best_model_weights.pth', map_location=device))
model.to(device)
model.eval()

# أسماء الفئات
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# تحسين واجهة Streamlit
st.set_page_config(page_title="Classification of brain tumors", layout="wide")

# إضافة Sidebar
st.sidebar.title("About")
st.sidebar.info("This application uses a deep learning model to classify brain tumors with high accuracy.")
with st.sidebar:

    with st.expander("What is a brain tumor?"):
        st.write("""
        A brain tumor is a collection, or mass, of abnormal cells in your brain. 
        Your skull, which encloses your brain, is very rigid. Any growth inside 
        such a restricted space can cause problems. Brain tumors can be cancerous 
        (malignant) or noncancerous (benign). When benign or malignant tumors grow, 
        they can cause the pressure inside your skull to increase. This can cause 
        brain damage, and it can be life-threatening.
        """)

    with st.expander("The importance of the subject"):
        st.write("""
        Early detection of brain tumors can significantly improve treatment outcomes. 
        Machine learning models can assist in rapid and accurate diagnosis, 
        helping doctors make better decisions.
        """)

# العنوان الرئيسي
st.title("🧠 Classification of brain tumors")

# رفع الصورة والتنبؤ
uploaded_file = st.file_uploader("📷 **Choose an image to analyze**", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📌 الصورة المرفوعة", use_column_width=True)
    
    # تجهيز الصورة للنموذج
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # عمل التنبؤ
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
    with col2:
        st.success(f"✅ **التصنيف المتوقع: {predicted_class}**")
        
    



