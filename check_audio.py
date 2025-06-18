import sounddevice as sd
import numpy as np

print("--- BẮT ĐẦU KIỂM TRA THIẾT BỊ ÂM THANH ---")

try:
    # 1. In ra danh sách tất cả các thiết bị âm thanh mà sounddevice tìm thấy
    print("\n[THÔNG TIN THIẾT BỊ]")
    print(sd.query_devices())

    # 2. Lấy thông tin về thiết bị mặc định
    default_device_index = sd.default.device[1] # 0 là input, 1 là output
    default_device_info = sd.query_devices(default_device_index, 'output')
    print(f"\n[THIẾT BỊ MẶC ĐỊNH ĐANG ĐƯỢC CHỌN]")
    print(f"ID: {default_device_index}, Tên: {default_device_info['name']}")
    print("-" * 20)

    # 3. Tạo ra một tín hiệu âm thanh đơn giản (âm A, tần số 440 Hz)
    # Điều này để đảm bảo chúng ta không phụ thuộc vào file audio của bạn
    fs = 44100  # Tần số lấy mẫu
    duration = 3  # 3 giây
    frequency = 440.0  # Tần số nốt A4
    
    print(f"\nĐang chuẩn bị phát một âm thanh thử nghiệm ({frequency} Hz) trong {duration} giây...")
    myarray = np.sin(2 * np.pi * frequency * np.arange(fs * duration) / fs)
    # Đảm bảo định dạng là float32, là định dạng phổ biến cho soundcard
    myarray = myarray.astype(np.float32)

    # 4. Phát âm thanh
    sd.play(myarray, fs)
    sd.wait()  # Chờ cho đến khi phát xong

    print("\nPhát âm thanh thử nghiệm thành công!")

except Exception as e:
    print(f"\n!!! ĐÃ XẢY RA LỖI !!!")
    print(f"Lỗi: {e}")

print("\n--- KẾT THÚC KIỂM TRA ---")
