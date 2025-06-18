import os
import re
import torch
from tkinter import messagebox

# Tải mô hình Silero VAD
def load_vad_model():
    """Tải mô hình Silero VAD và các tiện ích."""
    try:
        # torch.hub.set_dir('.') # Tùy chọn: đổi thư mục cache của torch hub
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        (get_speech_timestamps, _, read_audio, _, _) = utils
        return model, get_speech_timestamps, read_audio
    except Exception as e:
        messagebox.showerror("Lỗi tải mô hình", f"Không thể tải mô hình Silero VAD...\nLỗi: {e}")
        return None, None, None

# Tìm số thứ tự lớn nhất để đếm tiếp
def get_start_index(working_dir, prefix):
    """Quét thư mục để tìm chỉ số file lớn nhất và trả về chỉ số tiếp theo."""
    start_index = 1
    try:
        pattern = re.compile(f"^{re.escape(prefix)}_(\d+)\.wav$")
        existing_indices = []
        for filename in os.listdir(working_dir):
            match = pattern.match(filename)
            if match:
                existing_indices.append(int(match.group(1)))
        if existing_indices:
            start_index = max(existing_indices) + 1
    except Exception as e:
        print(f"Lỗi khi quét thư mục {working_dir}: {e}")
    return start_index
