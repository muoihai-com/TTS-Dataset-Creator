import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr
import threading
import os
import csv
from io import BytesIO
import numpy as np
import re # Thư viện để xử lý biểu thức chính quy, dùng để tách số từ tên file

# --- Hằng số và giá trị mặc định ---
class AppConfig:
    VAD_THRESHOLD = 0.4
    VAD_MIN_SILENCE_MS = 700
    VAD_MIN_SPEECH_MS = 250
    VAD_SPEECH_PAD_MS = 300
    EXPORT_SAMPLE_RATE = 24000
    EXPORT_CHANNELS = 1
    DEFAULT_PREFIX = "audio"

# --- Tải mô hình Silero VAD ---
def load_vad_model():
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        (get_speech_timestamps, _, read_audio, _, _) = utils
        return model, get_speech_timestamps, read_audio
    except Exception as e:
        messagebox.showerror("Lỗi tải mô hình", f"Không thể tải mô hình Silero VAD...\nLỗi: {e}")
        return None, None, None

class AudioProcessorApp(ctk.CTk):
    def __init__(self, model, get_speech_timestamps, read_audio):
        super().__init__()

        self.audio_file_path = None
        self.working_dir = None
        self.audio_segments_data = []
        self.vad_model, self.get_speech_timestamps, self.read_audio = model, get_speech_timestamps, read_audio
        self.title("Công cụ tạo Dataset TTS - Phiên bản Hoàn thiện")
        self.geometry("1000x800")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Giao diện người dùng ---
        self.setup_frame = ctk.CTkFrame(self)
        self.setup_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.setup_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.setup_frame, text="Bước 1:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(10,0), pady=10, sticky="w")
        self.select_button = ctk.CTkButton(self.setup_frame, text="Chọn file Audio nguồn", command=self.select_audio_file)
        self.select_button.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.file_label = ctk.CTkLabel(self.setup_frame, text="Chưa chọn file", anchor="w")
        self.file_label.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(self.setup_frame, text="Bước 2:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=(10,0), pady=10, sticky="w")
        self.select_dir_button = ctk.CTkButton(self.setup_frame, text="Chọn Thư mục làm việc", command=self.select_working_dir, state="disabled")
        self.select_dir_button.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.dir_label = ctk.CTkLabel(self.setup_frame, text="Chưa chọn thư mục", anchor="w")
        self.dir_label.grid(row=1, column=2, padx=10, pady=10, sticky="ew")
        
        # [TÍNH NĂNG MỚI] Ô nhập tiền tố
        ctk.CTkLabel(self.setup_frame, text="Tiền tố tên file (tùy chọn):").grid(row=2, column=0, padx=(10,0), pady=10, sticky="w")
        self.prefix_entry = ctk.CTkEntry(self.setup_frame, placeholder_text=f"Mặc định: '{AppConfig.DEFAULT_PREFIX}'")
        self.prefix_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.params_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.params_frame.grid(row=1, column=0, padx=10, pady=0, sticky="ew")
        self.vad_params = {}
        params_to_show = {
            'threshold': ("Ngưỡng", AppConfig.VAD_THRESHOLD), 'min_silence_duration_ms': ("Khoảng lặng", AppConfig.VAD_MIN_SILENCE_MS),
            'min_speech_duration_ms': ("Giọng nói", AppConfig.VAD_MIN_SPEECH_MS), 'speech_pad_ms': ("Vùng đệm", AppConfig.VAD_SPEECH_PAD_MS)
        }
        for i, (key, (label, value)) in enumerate(params_to_show.items()):
            ctk.CTkLabel(self.params_frame, text=f"{label} (ms):").grid(row=0, column=i*2, padx=(10, 5), pady=5)
            entry = ctk.CTkEntry(self.params_frame, width=80)
            entry.insert(0, str(value))
            entry.grid(row=0, column=i*2+1, padx=(0, 20), pady=5)
            self.vad_params[key] = entry
        
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.process_button = ctk.CTkButton(self.control_frame, text="3. Bắt đầu Xử lý", command=self.start_processing_thread, state="disabled")
        self.process_button.pack(side="left", padx=10, pady=10)
        self.noise_reduce_checkbox = ctk.CTkCheckBox(self.control_frame, text="Áp dụng Khử nhiễu")
        self.noise_reduce_checkbox.pack(side="left", padx=10, pady=10)
        self.noise_reduce_checkbox.deselect()
        self.status_label = ctk.CTkLabel(self.control_frame, text="Sẵn sàng")
        self.status_label.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Danh sách Audio đã xử lý")
        self.scrollable_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        self.clear_all_button = ctk.CTkButton(self.bottom_frame, text="Xóa sạch file đã tạo trong Thư mục làm việc", command=self.clear_working_directory, hover_color="#C82333")
        self.clear_all_button.pack(side="left", padx=10, pady=10)
        self.save_metadata_button = ctk.CTkButton(self.bottom_frame, text="4. Lưu file metadata.csv", command=self.save_metadata, state="disabled")
        self.save_metadata_button.pack(side="right", padx=10, pady=10)

    # --- Các hàm chức năng ---
    def select_audio_file(self):
        filepath = filedialog.askopenfilename(title="Bước 1: Chọn một file audio", filetypes=(("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")))
        if filepath:
            self.audio_file_path = filepath
            self.file_label.configure(text=os.path.basename(filepath))
            self.select_dir_button.configure(state="normal")
            if self.working_dir:
                self.process_button.configure(state="normal")
                self.clear_ui_list()

    def select_working_dir(self):
        dirpath = filedialog.askdirectory(title="Bước 2: Chọn một Thư mục làm việc để lưu kết quả")
        if dirpath:
            self.working_dir = dirpath
            self.dir_label.configure(text=dirpath)
            if self.audio_file_path:
                self.process_button.configure(state="normal")
            self.clear_ui_list()

    def clear_ui_list(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.audio_segments_data = []
        self.save_metadata_button.configure(state="disabled")

    def start_processing_thread(self):
        if not self.audio_file_path or not self.working_dir:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn file audio và thư mục làm việc trước.")
            return
        try:
            self.current_vad_params = { key: float(entry.get()) for key, entry in self.vad_params.items() }
            self.current_vad_params['min_silence_duration_ms'] = int(self.current_vad_params['min_silence_duration_ms'])
            self.current_vad_params['min_speech_duration_ms'] = int(self.current_vad_params['min_speech_duration_ms'])
            self.current_vad_params['speech_pad_ms'] = int(self.current_vad_params['speech_pad_ms'])
        except ValueError:
            messagebox.showerror("Lỗi tham số", "Các tham số VAD phải là số. Vui lòng kiểm tra lại.")
            return
        
        self.clear_ui_list()
        self.process_button.configure(state="disabled")
        self.select_button.configure(state="disabled")
        self.select_dir_button.configure(state="disabled")
        self.status_label.configure(text="Đang xử lý, vui lòng chờ...")
        
        thread = threading.Thread(target=self.process_and_save_audio)
        thread.start()

    # [LOGIC MỚI] Tìm số thứ tự lớn nhất để đếm tiếp
    def get_start_index(self, prefix):
        start_index = 1
        try:
            # Biểu thức chính quy để tìm số trong tên file, ví dụ: 'abc_001.wav' -> '001'
            pattern = re.compile(f"^{re.escape(prefix)}_(\d+)\.wav$")
            existing_indices = []
            for filename in os.listdir(self.working_dir):
                match = pattern.match(filename)
                if match:
                    existing_indices.append(int(match.group(1)))
            if existing_indices:
                start_index = max(existing_indices) + 1
        except Exception as e:
            print(f"Lỗi khi quét thư mục: {e}")
        return start_index

    def process_and_save_audio(self):
        try:
            prefix = self.prefix_entry.get().strip() or AppConfig.DEFAULT_PREFIX
            start_index = self.get_start_index(prefix)
            current_file_index = 0

            main_audio = AudioSegment.from_file(self.audio_file_path)
            temp_wav_io = BytesIO()
            main_audio.export(temp_wav_io, format="wav")
            temp_wav_io.seek(0)
            wav_tensor = self.read_audio(temp_wav_io, sampling_rate=16000)
            
            speech_timestamps = self.get_speech_timestamps(wav_tensor, self.vad_model, sampling_rate=16000, **self.current_vad_params)
            
            for ts in speech_timestamps:
                segment = main_audio[ts['start']*1000/16000 : ts['end']*1000/16000]
                samples_int = np.array(segment.get_array_of_samples())
                
                if self.noise_reduce_checkbox.get():
                    reduced_float = nr.reduce_noise(y=samples_int, sr=segment.frame_rate)
                    samples_for_export = (reduced_float * 32767).astype(np.int16)
                else:
                    samples_for_export = samples_int

                processed_segment = AudioSegment(samples_for_export.tobytes(), sample_width=2, frame_rate=segment.frame_rate, channels=segment.channels)
                final_segment = processed_segment.set_frame_rate(AppConfig.EXPORT_SAMPLE_RATE).set_channels(AppConfig.EXPORT_CHANNELS)
                
                # [LOGIC MỚI] Sử dụng tiền tố và chỉ số bắt đầu
                filename = f"{prefix}_{start_index + current_file_index:03d}.wav"
                export_path = os.path.join(self.working_dir, filename)
                final_segment.export(export_path, format="wav")
                
                self.audio_segments_data.append({
                    'id': start_index + current_file_index, 
                    'path': export_path,
                    'duration': len(final_segment) / 1000.0
                })
                current_file_index += 1

            self.after(0, self.update_ui_with_segments)
        except Exception as e:
            error_msg = f"Đã xảy ra lỗi trong quá trình xử lý:\n{e}"
            self.after(0, lambda: messagebox.showerror("Lỗi xử lý", error_msg))
        finally:
            self.after(0, self.processing_finished)

    def update_ui_with_segments(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        if not self.audio_segments_data:
            self.status_label.configure(text="Không tìm thấy giọng nói trong file.")
            return
        
        sorted_segments = sorted(self.audio_segments_data, key=lambda x: x['id'])
        self.audio_segments_data = sorted_segments

        for i, segment_data in enumerate(self.audio_segments_data):
            segment_frame = ctk.CTkFrame(self.scrollable_frame)
            segment_frame.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            segment_frame.grid_columnconfigure(2, weight=1)

            info_text = f"{os.path.basename(segment_data['path'])} ({segment_data['duration']:.2f}s)"
            filename_label = ctk.CTkLabel(segment_frame, text=info_text, width=150)
            filename_label.grid(row=0, column=0, padx=5, pady=5)
            
            play_button = ctk.CTkButton(
                segment_frame, text="▶ Play", width=80,
                command=lambda path=segment_data['path']: threading.Thread(target=self.play_audio_file, args=(path,), daemon=True).start()
            )
            play_button.grid(row=0, column=1, padx=5, pady=5)
            
            transcript_entry = ctk.CTkEntry(segment_frame, placeholder_text="Nhập transcript tại đây...")
            transcript_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

            delete_button = ctk.CTkButton(
                segment_frame, text="🗑️", width=30, fg_color="transparent", hover_color="#C82333",
                command=lambda seg_data=segment_data: self.delete_segment(seg_data)
            )
            delete_button.grid(row=0, column=3, padx=5, pady=5)
            
            segment_data.update({'ui_frame': segment_frame, 'entry_widget': transcript_entry})

    def processing_finished(self):
        current_status = self.status_label.cget("text")
        if "Đang xử lý" in current_status:
             self.status_label.configure(text=f"Hoàn tất! Đã tạo {len(self.audio_segments_data)} file mới.")
        
        self.process_button.configure(state="normal")
        self.select_button.configure(state="normal")
        self.select_dir_button.configure(state="normal")
        if self.audio_segments_data:
            self.save_metadata_button.configure(state="normal")

    def play_audio_file(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            play(audio)
        except Exception as e:
            print(f"Lỗi khi phát file {os.path.basename(file_path)}: {e}")

    def delete_segment(self, segment_data):
        if messagebox.askyesno("Xác nhận Xóa", f"Bạn có chắc muốn xóa vĩnh viễn file:\n{os.path.basename(segment_data['path'])}?"):
            try:
                os.remove(segment_data['path'])
                segment_data['ui_frame'].destroy()
                self.audio_segments_data.remove(segment_data)
            except Exception as e: messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")
            
    # [LOGIC MỚI] Nối tiếp dữ liệu vào file CSV
    def save_metadata(self):
        if not self.working_dir:
            messagebox.showerror("Lỗi", "Không tìm thấy thư mục làm việc.")
            return
            
        csv_path = os.path.join(self.working_dir, "metadata.csv")
        file_exists = os.path.isfile(csv_path)
        
        try:
            # Mở file ở chế độ 'a' (append) để nối tiếp
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Chỉ ghi header nếu file chưa tồn tại
                if not file_exists:
                    writer.writerow(['audio_filename', 'transcript'])
                
                # Chỉ ghi những dòng mới được tạo trong phiên này
                for segment_data in self.audio_segments_data:
                    filename = os.path.basename(segment_data['path'])
                    transcript = segment_data['entry_widget'].get().lower()
                    if transcript:
                        writer.writerow([filename, transcript])
            
            messagebox.showinfo("Hoàn tất", f"Đã lưu/cập nhật thành công file metadata.csv tại:\n{self.working_dir}")
            # Sau khi lưu, xóa danh sách trong bộ nhớ để tránh lưu trùng lặp
            self.clear_ui_list()
            self.status_label.configure(text="Đã lưu metadata. Sẵn sàng cho phiên mới.")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể ghi file CSV: {e}")

    def clear_working_directory(self):
        if not self.working_dir or not os.path.exists(self.working_dir):
            messagebox.showinfo("Thông báo", "Vui lòng chọn một thư mục làm việc hợp lệ trước.")
            return
        
        prefix = self.prefix_entry.get().strip() or AppConfig.DEFAULT_PREFIX
        warning_message = (f"Hành động này sẽ XÓA TẤT CẢ các file có tên dạng '{prefix}_*.wav' khỏi thư mục:\n\n{self.working_dir}\n\nBạn có chắc chắn không?")
        
        if messagebox.askyesno("CẢNH BÁO XÓA FILE", warning_message):
            files_deleted_count = 0
            try:
                for filename in os.listdir(self.working_dir):
                    if filename.startswith(f"{prefix}_") and filename.endswith('.wav'):
                        try:
                            os.remove(os.path.join(self.working_dir, filename))
                            files_deleted_count += 1
                        except Exception as file_error: print(f"Không thể xóa file {filename}: {file_error}")
                
                self.clear_ui_list()
                self.status_label.configure(text=f"Đã xóa {files_deleted_count} file có tiền tố '{prefix}' khỏi thư mục.")
                messagebox.showinfo("Hoàn tất", f"Đã xóa thành công {files_deleted_count} file audio.")

            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi dọn dẹp thư mục: {e}")

if __name__ == "__main__":
    print("Đang tải mô hình Silero VAD, vui lòng chờ...")
    model, get_speech_timestamps, read_audio = load_vad_model()
    if model:
        print("Tải mô hình thành công. Đang khởi động ứng dụng...")
        app = AudioProcessorApp(model, get_speech_timestamps, read_audio)
        app.mainloop()
    else:
        print("Không thể khởi động ứng dụng do lỗi tải mô hình.")
