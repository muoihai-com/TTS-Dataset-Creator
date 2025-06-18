# Nhiệm vụ duy nhất của file này là khởi chạy ứng dụng

import customtkinter as ctk
from app_gui import AudioProcessorApp
from audio_utils import load_vad_model

if __name__ == "__main__":
    print("Đang tải mô hình Silero VAD, vui lòng chờ...")
    
    # Tải mô hình trước khi khởi tạo giao diện
    model, get_speech_timestamps, read_audio = load_vad_model()
    
    # Chỉ khởi chạy app nếu mô hình được tải thành công
    if model:
        print("Tải mô hình thành công. Đang khởi động ứng dụng...")
        app = AudioProcessorApp(model, get_speech_timestamps, read_audio)
        app.mainloop()
    else:
        print("Không thể khởi động ứng dụng do lỗi tải mô hình.")
        # Hiển thị một cửa sổ lỗi đơn giản nếu không có GUI
        root = ctk.CTk()
        root.withdraw()
        ctk.messagebox.showerror("Lỗi nghiêm trọng", "Không thể tải mô hình VAD. Vui lòng kiểm tra kết nối mạng và thử lại.")
        root.destroy()
