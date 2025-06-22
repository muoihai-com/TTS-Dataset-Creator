import os
from datasets import load_dataset
import soundfile as sf

def save_vietnamese_voices_dataset(output_dir="wavs"):
    """
    Tải bộ dữ liệu 'hungkieu/vietnamese-voices' và lưu các tệp âm thanh
    vào một thư mục được chỉ định, xử lý đúng cấu trúc thư mục con.

    Args:
        output_dir (str): Tên thư mục gốc để lưu các tệp âm thanh.
    """
    # Tên bộ dữ liệu trên Hugging Face Hub
    dataset_name = "hungkieu/vietnamese-voices"

    # Tạo thư mục gốc nếu nó chưa tồn tại (chỉ cần tạo thư mục wavs ban đầu)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục gốc: {output_dir}")

    try:
        # Tải bộ dữ liệu
        print(f"Đang tải bộ dữ liệu '{dataset_name}'...")
        ds = load_dataset(dataset_name, split='train', streaming=True)
        print("Tải bộ dữ liệu thành công.")

        # Lặp qua từng mẫu trong bộ dữ liệu và lưu tệp âm thanh
        print(f"Bắt đầu lưu các tệp âm thanh vào thư mục '{output_dir}'...")
        saved_count = 0
        for example in ds:
            # Lấy dữ liệu âm thanh và tốc độ lấy mẫu
            audio_data = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]

            # Lấy tên tệp âm thanh từ cột 'audio_name'
            audio_name = example["audio_filename"]
            
            # --- PHẦN SỬA LỖI ---
            # Tách audio_name thành các phần dựa trên ký tự '/'
            path_parts = audio_name.split('/')
            
            # Nối các phần lại bằng ký tự phân cách của hệ điều hành hiện tại
            # để tạo đường dẫn đầy đủ và chính xác
            output_path = os.path.join(output_dir, *path_parts)

            # Lấy đường dẫn thư mục chứa tệp (ví dụ: "wavs/nam")
            output_file_dir = os.path.dirname(output_path)

            # Tạo các thư mục con nếu chúng chưa tồn tại (ví dụ: tạo thư mục 'nam')
            os.makedirs(output_file_dir, exist_ok=True)
            # --- KẾT THÚC PHẦN SỬA LỖI ---

            # Đảm bảo tên tệp có đuôi .wav
            if not output_path.lower().endswith('.wav'):
                output_path += '.wav'

            # Lưu mảng numpy thành tệp .wav
            sf.write(output_path, audio_data, sampling_rate, subtype='PCM_16')
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"Đã lưu {saved_count} tệp...")

        print(f"\nHoàn tất! Đã lưu thành công {saved_count} tệp vào thư mục '{output_dir}' với cấu trúc thư mục con chính xác.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        print("Vui lòng kiểm tra lại bạn đã đăng nhập vào Hugging Face (huggingface-cli login) và có kết nối internet.")

if __name__ == "__main__":
    save_vietnamese_voices_dataset()
