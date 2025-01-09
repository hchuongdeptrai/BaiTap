import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

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
        # Tính toán padding dựa trên kích thước khuôn mặt
        padding = int(min(w, h) * 0.1)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])
        
        face_region = image[y1:y2, x1:x2]
        
        # Tính kernel size dựa trên độ mờ và kích thước khuôn mặt
        # Đảm bảo kernel size tỷ lệ với cả độ mờ và kích thước khuôn mặt
        base_kernel = int((blur_strength / 100.0) * min(w, h))  # Kernel size tỷ lệ với độ mờ và kích thước mặt
        kernel_size = max(base_kernel, 3)  # Đảm bảo kernel size tối thiểu là 3
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Đảm bảo kernel size là số lẻ
        
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        # Tạo mặt nạ hình elip và làm mịn viền
        mask = create_elliptical_mask(y2-y1, x2-x1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)  # Làm mịn viền mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Kết hợp vùng gốc và vùng làm mờ
        face_region_new = face_region * (1 - mask_3channel) + blurred_face * mask_3channel
        image[y1:y2, x1:x2] = face_region_new
    
    return image

def process_video(video_path, blur_strength, progress_bar):
    # Đọc video input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo file output tạm thời
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    # Khởi tạo face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Xử lý từng frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Xử lý và làm mờ khuôn mặt trong frame hiện tại
        frame_processed = blur_face(frame, face_cascade, blur_strength)
        
        # Ghi frame đã xử lý
        out.write(frame_processed)
        
        # Cập nhật tiến trình
        frame_count += 1
        if frame_count % 10 == 0:  # Cập nhật progress bar mỗi 10 frame
            progress = frame_count / total_frames
            progress_bar.progress(progress, f"Processing: {frame_count}/{total_frames} frames")
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
    return temp_output

def main():
    st.title("Ứng dụng Phát hiện và Làm mờ Khuôn mặt")
    
    st.write("Tải lên ảnh hoặc video để phát hiện và làm mờ khuôn mặt.")
    
    blur_strength = st.slider(
        "Điều chỉnh độ mờ",
        min_value=1,
        max_value=100,
        value=50,
        help="Kéo thanh trượt để điều chỉnh độ mờ của khuôn mặt"
    )
    
    mode = st.radio("Chọn chế độ:", ["Ảnh", "Video"])
    
    if mode == "Ảnh":
        uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.subheader("Ảnh gốc")
            st.image(image, use_column_width=True)
            
            if st.button('Phát hiện và Làm mờ Khuôn mặt'):
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                processed_image = blur_face(image, face_cascade, blur_strength)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                st.subheader("Ảnh sau khi xử lý")
                st.image(processed_image, use_column_width=True)
                
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                io_buf = io.BytesIO(buffer)
                
                btn = st.download_button(
                    label="Tải ảnh đã xử lý",
                    data=io_buf.getvalue(),
                    file_name="blurred_face.png",
                    mime="image/png"
                )
    else:
        uploaded_file = st.file_uploader("Chọn video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.video(uploaded_file)
            
            if st.button('Phát hiện và Làm mờ Khuôn mặt'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner('Đang xử lý video...'):
                    output_video = process_video(tfile.name, blur_strength, progress_bar)
                    
                progress_bar.empty()
                status_text.empty()
                
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

if __name__ == "__main__":
    main()