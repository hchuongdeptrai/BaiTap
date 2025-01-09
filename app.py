import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

def show_theory():
    st.header("1. Phương pháp phát hiện và làm mờ khuôn mặt")
    
    st.subheader("1.1. Phát hiện khuôn mặt bằng Haar Cascade")
    st.write("""
    Haar Cascade là một phương pháp học máy để phát hiện đối tượng được phát triển bởi Paul Viola và Michael Jones. 
    Phương pháp này sử dụng các đặc trưng Haar để phát hiện khuôn mặt trong hình ảnh:
    
    Có thể minh họa quá trình phát hiện gương mặt như sau :

    - (1) Ảnh dầu vào được chia thành nhiều cửa sổ nhỏ (subwindows) để quét toàn bộ hình ảnh.
    - (2) Các cửa sổ này sẽ được duyệt tuần tự để kiểm tra sự hiện diện của khuôn mặt.
    - (3) Với từng cửa sổ con, đặc trưng quan trọng sẽ được trích xuất.
    - (4) Sau đó, các đặc trưng của từng cửa sổ con được so sánh với vector đặc trưng của tập huấn luyện.
    - (5) Sử dụng cơ chế cơ chế "Cascade of Classifiers" (Phân loại theo tầng) để phát hiện gương mặt.
    - (6) Nếu cửa sổ con được xác định là gương mặt, thì sẽ được phân loại là gương mặt, còn không thì ngược lại.
    """)

    col1 , col2 , col3 = st.columns([1,10,1])
    with col2:
        st.image("Ảnh Gốc.jpg", caption="Minh họa quá trình phat hiện khuôn mặt bằng Haar Cascade")

    st.subheader("1.2. Làm mờ khuôn mặt bằng Gaussian Blur")
    st.write("""
    Gaussian Blur là một kỹ thuật làm mờ hình ảnh dựa trên phân phối Gaussian:

    1. **Nguyên lý hoạt động**:
    - Sử dụng ma trận kernel có trọng số theo phân phối Gaussian 2D
    - Mỗi pixel mới là trung bình có trọng số của các pixel lân cận
    - Trọng số giảm dần theo khoảng cách từ pixel trung tâm

    2. **Tham số quan trọng**:
    - Kích thước kernel: Quyết định mức độ làm mờ
    - Độ lệch chuẩn (σ): Kiểm soát hình dạng của phân phối Gaussian

    3. **Ưu điểm**:
    - Làm mờ tự nhiên, mượt mà
    - Giảm nhiễu hiệu quả
    - Bảo toàn các cạnh tốt hơn so với làm mờ trung bình

    4. **Áp dụng trong ứng dụng**:
    - Kernel size tỷ lệ với kích thước khuôn mặt
    - Sử dụng mặt nạ hình elip để tạo hiệu ứng tự nhiên
    - Làm mịn viền để tránh hiện tượng răng cưa
    """)

    # Thêm hai cột để hiển thị hình ảnh và gif minh họa
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("gaussian_kernel_3d.png", caption="Ảnh gốc")
    
    with col2:
        st.image("gaussian_blur_demo.png", caption="Minh họa quá trình áp dụng Gaussian Blur với 1 phần nhỏ")

    st.markdown("Minh họa")

    col1, col2 = st.columns(2)
    
    with col1:
        st.image("cat.jpg", caption="Minh họa quá trình áp dụng Gaussian Blur với 1 phần nghiệm")
    
    with col2:
        st.image("blur-demo.gif", caption="Minh họa quá trình áp dụng Gaussian Blur với 1 phần lớn")

   
def show_application():
    st.write("Tải lên ảnh hoặc video để phát hiện và làm mờ khuôn mặt.")
    
    # Tạo container cho phần điều khiển
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            blur_strength = st.slider(
                "Điều chỉnh độ mờ",
                min_value=1,
                max_value=100,
                value=50,
                help="Kéo thanh trượt để điều chỉnh độ mờ (1: mờ nhẹ, 100: mờ tối đa)"
            )
        
        with col2:
            mode = st.radio("Chọn chế độ:", ["Ảnh", "Video"])
    
    # Container cho phần upload và hiển thị
    with st.container():
        if mode == "Ảnh":
            uploaded_file = st.file_uploader(
                "Chọn ảnh...", 
                type=['jpg', 'jpeg', 'png'],
                help="Kéo thả hoặc chọn file ảnh từ máy tính"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Ảnh gốc")
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                    if st.button('Phát hiện và làm mờ khuôn mặt'):
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        processed_image = blur_face(image, face_cascade, blur_strength)
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.subheader("Ảnh sau khi xử lý")
                            st.image(processed_image, use_column_width=True)
                            
                            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                            io_buf = io.BytesIO(buffer)
                            
                            st.download_button(
                                label="Tải ảnh đã xử lý",
                                data=io_buf.getvalue(),
                                file_name="blurred_face.png",
                                mime="image/png"
                            )
        else:
            uploaded_file = st.file_uploader(
                "Chọn video...", 
                type=['mp4', 'avi', 'mov'],
                help="Kéo thả hoặc chọn file video từ máy tính"
            )
            
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Video gốc")
                    st.video(uploaded_file)
                    if st.button('Phát hiện và làm mờ khuôn mặt'):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner('Đang xử lý video...'):
                            output_video = process_video(tfile.name, blur_strength, progress_bar)
                            
                        progress_bar.empty()
                        status_text.empty()
                        
                        with col2:
                            st.subheader("Video sau khi xử lý")
                            st.video(output_video)
                            
                            with open(output_video, 'rb') as f:
                                video_bytes = f.read()
                                st.download_button(
                                    label="Tải video đã xử lý",
                                    data=video_bytes,
                                    file_name="blurred_face.mp4",
                                    mime="video/mp4"
                                )

def create_elliptical_mask(height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (19, 19), 0)
    return mask

def blur_face(image, face_cascade, blur_strength):
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        padding = int(min(w, h) * 0.1)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])
        
        face_region = image[y1:y2, x1:x2]
        
        base_kernel = int((blur_strength / 100.0) * min(w, h))
        kernel_size = max(base_kernel, 3)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        mask = create_elliptical_mask(y2-y1, x2-x1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        face_region_new = face_region * (1 - mask_3channel) + blurred_face * mask_3channel
        image[y1:y2, x1:x2] = face_region_new
    
    return image

def process_video(video_path, blur_strength, progress_bar):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_processed = blur_face(frame, face_cascade, blur_strength)
        out.write(frame_processed)
        
        frame_count += 1
        if frame_count % 10 == 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress, f"Processing: {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    return temp_output

def main():
    st.set_page_config(
        page_title="Ứng dụng làm mờ khuôn mặt trong hình ảnh hoặc video giám sát",
        page_icon="👤",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS để tùy chỉnh giao diện
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            text-align: center;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.title("Ứng dụng Phát hiện và Làm mờ Khuôn mặt")
        
        # Tạo tabs
        tab1, tab2 = st.tabs(["Lý thuyết", "Ứng dụng"])
        
        with tab1:
            show_theory()
        
        with tab2:
            show_application()

if __name__ == "__main__":
    main()
