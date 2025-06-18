import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr
import threading
import os
import csv
from io import BytesIO
import numpy as np
import re

# Import từ các file mới
from config import AppConfig
from audio_utils import get_start_index

class AudioProcessorApp(ctk.CTk):
    def __init__(self, model, get_speech_timestamps, read_audio):
        super().__init__()

        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.wavs_dir_path = os.path.join(self.project_dir, AppConfig.WAVS_DIR)
        self.metadata_path = os.path.join(self.project_dir, AppConfig.METADATA_FILE)
        os.makedirs(self.wavs_dir_path, exist_ok=True)

        self.audio_file_path = None
        self.current_speaker = None
        self.audio_segments_data = []
        
        self.vad_model, self.get_speech_timestamps, self.read_audio = model, get_speech_timestamps, read_audio
        self.title("Trình quản lý Dataset TTS")
        self.geometry("1100x800")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Giao diện người dùng ---
        self.setup_frame = ctk.CTkFrame(self)
        self.setup_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.setup_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.setup_frame, text="Bước 1:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(10,0), pady=5, sticky="w")
        self.select_button = ctk.CTkButton(self.setup_frame, text="Chọn file Audio nguồn", command=self.select_audio_file)
        self.select_button.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.file_label = ctk.CTkLabel(self.setup_frame, text="Chưa chọn file", anchor="w")
        self.file_label.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.setup_frame, text="Bước 2:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=(10,0), pady=5, sticky="w")
        self.speaker_selector = ctk.CTkComboBox(self.setup_frame, command=self.load_speaker_session, width=200)
        self.speaker_selector.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.new_speaker_button = ctk.CTkButton(self.setup_frame, text="Tạo người nói mới", width=140, command=self.create_new_speaker)
        self.new_speaker_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        
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
        self.process_button = ctk.CTkButton(self.control_frame, text="3. Xử lý & Thêm Audio", command=self.start_processing_thread, state="disabled")
        self.process_button.pack(side="left", padx=10, pady=10)
        self.noise_reduce_checkbox = ctk.CTkCheckBox(self.control_frame, text="Áp dụng Khử nhiễu")
        self.noise_reduce_checkbox.pack(side="left", padx=10, pady=10)
        
        self.status_label = ctk.CTkLabel(self.control_frame, text="Chào mừng!")
        self.status_label.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Danh sách Audio")
        self.scrollable_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        self.clear_all_button = ctk.CTkButton(self.bottom_frame, text="Xóa các file của người nói này", command=self.clear_speaker_directory, hover_color="#C82333", state="disabled")
        self.clear_all_button.pack(side="left", padx=10, pady=10)
        self.save_metadata_button = ctk.CTkButton(self.bottom_frame, text="4. Lưu tất cả thay đổi", command=self.save_metadata, state="disabled")
        self.save_metadata_button.pack(side="right", padx=10, pady=10)

        self.update_speaker_list()

    def update_speaker_list(self):
        try:
            speakers = [d for d in os.listdir(self.wavs_dir_path) if os.path.isdir(os.path.join(self.wavs_dir_path, d))]
            current_selection = self.speaker_selector.get()
            self.speaker_selector.configure(values=speakers if speakers else ["(Chưa có người nói)"])
            
            if current_selection in speakers:
                self.speaker_selector.set(current_selection)
            elif speakers:
                self.speaker_selector.set(speakers[0])
                self.load_speaker_session(speakers[0])
            else:
                self.speaker_selector.set("")
        except Exception as e:
            print(f"Lỗi khi cập nhật danh sách người nói: {e}")

    def create_new_speaker(self):
        speaker_name = simpledialog.askstring("Tạo người nói mới", "Nhập tên cho người nói mới (ví dụ: speaker_03):")
        if speaker_name and speaker_name.strip():
            speaker_name = speaker_name.strip().replace(" ", "_")
            new_speaker_path = os.path.join(self.wavs_dir_path, speaker_name)
            if not os.path.exists(new_speaker_path):
                os.makedirs(new_speaker_path)
                messagebox.showinfo("Thành công", f"Đã tạo thư mục cho người nói: {speaker_name}")
                self.update_speaker_list()
                self.speaker_selector.set(speaker_name)
                self.load_speaker_session(speaker_name)
            else:
                messagebox.showwarning("Tồn tại", "Người nói này đã tồn tại.")
                self.speaker_selector.set(speaker_name)

    def load_speaker_session(self, speaker_name):
        if not speaker_name or speaker_name == "(Chưa có người nói)":
            self.clear_ui_list()
            self.process_button.configure(state="disabled")
            self.clear_all_button.configure(state="disabled")
            self.current_speaker = None
            return
        
        self.current_speaker = speaker_name
        self.clear_ui_list()
        self.clear_all_button.configure(state="normal")
        if self.audio_file_path: self.process_button.configure(state="normal")

        all_metadata = []
        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                all_metadata = list(reader)

        # [SỬA LỖI] Logic đọc file CSV theo định dạng mới
        expected_parent_dir = self.current_speaker
        
        loaded_segments = []
        for row in all_metadata:
            try:
                # Lấy thư mục cha từ đường dẫn trong file CSV
                file_parent_dir = os.path.dirname(row['audio_filename'])
                
                if file_parent_dir == expected_parent_dir:
                    # Đường dẫn vật lý đầy đủ
                    full_path = os.path.join(self.wavs_dir_path, row['audio_filename'])
                    
                    if os.path.exists(full_path):
                        duration = 0.0
                        try:
                            audio = AudioSegment.from_file(full_path)
                            duration = len(audio) / 1000.0
                        except Exception as e:
                            print(f"Không thể đọc file audio: {full_path}, lỗi: {e}")

                        file_id_match = re.search(r'_(\d+)\.wav$', os.path.basename(full_path))
                        file_id = int(file_id_match.group(1)) if file_id_match else -1
                        
                        loaded_segments.append({
                            'id': file_id, 'path': full_path,
                            'duration': duration, 'transcript': row.get('transcript', '')
                        })
            except (KeyError, IndexError) as e:
                print(f"Bỏ qua dòng lỗi trong CSV: {row}, lỗi: {e}")
                continue

        self.audio_segments_data = loaded_segments
        self.status_label.configure(text=f"Đã tải {len(loaded_segments)} file của: {self.current_speaker}")
        self.update_ui_with_segments()

    def select_audio_file(self):
        filepath = filedialog.askopenfilename(title="Bước 1: Chọn file audio", filetypes=(("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")))
        if filepath:
            self.audio_file_path = filepath
            self.file_label.configure(text=os.path.basename(filepath))
            if self.current_speaker: self.process_button.configure(state="normal")

    def clear_ui_list(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.audio_segments_data = []
        self.save_metadata_button.configure(state="disabled")

    def start_processing_thread(self):
        if not self.audio_file_path or not self.current_speaker:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn file audio và một người nói trước.")
            return
        try:
            self.current_vad_params = { key: float(entry.get()) for key, entry in self.vad_params.items() }
            self.current_vad_params['min_silence_duration_ms'] = int(self.current_vad_params['min_silence_duration_ms'])
            self.current_vad_params['min_speech_duration_ms'] = int(self.current_vad_params['min_speech_duration_ms'])
            self.current_vad_params['speech_pad_ms'] = int(self.current_vad_params['speech_pad_ms'])
        except ValueError:
            messagebox.showerror("Lỗi tham số", "Các tham số VAD phải là số.")
            return
        
        # Xóa danh sách tạm thời trên UI nhưng không xóa dữ liệu đã load
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()

        self.process_button.configure(state="disabled")
        self.status_label.configure(text=f"Đang xử lý cho {self.current_speaker}, vui lòng chờ...")
        
        thread = threading.Thread(target=self.process_and_save_audio)
        thread.start()

    def process_and_save_audio(self):
        newly_created_segments = []
        try:
            speaker_path = os.path.join(self.wavs_dir_path, self.current_speaker)
            start_index = get_start_index(speaker_path, 'audio') # Prefix mặc định là 'audio'
            
            main_audio = AudioSegment.from_file(self.audio_file_path)
            wav_tensor = self.read_audio(BytesIO(main_audio.export(format="wav").read()), sampling_rate=16000)
            speech_timestamps = self.get_speech_timestamps(wav_tensor, self.vad_model, sampling_rate=16000, **self.current_vad_params)
            
            for i, ts in enumerate(speech_timestamps):
                segment = main_audio[ts['start']*1000/16000 : ts['end']*1000/16000]
                samples_int = np.array(segment.get_array_of_samples())
                
                if self.noise_reduce_checkbox.get():
                    reduced_float = nr.reduce_noise(y=samples_int, sr=segment.frame_rate)
                    samples_for_export = (reduced_float * 32767).astype(np.int16)
                else:
                    samples_for_export = samples_int

                processed_segment = AudioSegment(samples_for_export.tobytes(), sample_width=2, frame_rate=segment.frame_rate, channels=segment.channels)
                final_segment = processed_segment.set_frame_rate(AppConfig.EXPORT_SAMPLE_RATE).set_channels(AppConfig.EXPORT_CHANNELS)
                
                filename = f"audio_{start_index + i:03d}.wav"
                export_path = os.path.join(speaker_path, filename)
                final_segment.export(export_path, format="wav")
                
                newly_created_segments.append({
                    'id': start_index + i, 'path': export_path,
                    'duration': len(final_segment) / 1000.0, 'transcript': ''
                })

            self.audio_segments_data.extend(newly_created_segments)
            self.after(0, self.update_ui_with_segments)

        except Exception as e:
            error_msg = f"Đã xảy ra lỗi trong quá trình xử lý:\n{e}"
            self.after(0, lambda: messagebox.showerror("Lỗi xử lý", error_msg))
        finally:
            self.after(0, self.processing_finished, len(newly_created_segments))

    def update_ui_with_segments(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        if not self.audio_segments_data: return
        
        sorted_segments = sorted(self.audio_segments_data, key=lambda x: x['path'])
        self.audio_segments_data = sorted_segments
        
        for i, segment_data in enumerate(sorted_segments):
            segment_frame = ctk.CTkFrame(self.scrollable_frame)
            segment_frame.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            segment_frame.grid_columnconfigure(2, weight=1)

            info_text = f"{os.path.basename(os.path.dirname(segment_data['path']))}/{os.path.basename(segment_data['path'])} ({segment_data['duration']:.2f}s)"
            ctk.CTkLabel(segment_frame, text=info_text, width=200).grid(row=0, column=0, padx=5, pady=5)
            
            play_button = ctk.CTkButton(segment_frame, text="▶ Play", width=70, command=lambda path=segment_data['path']: threading.Thread(target=self.play_audio_file, args=(path,), daemon=True).start())
            play_button.grid(row=0, column=1, padx=5, pady=5)
            
            entry = ctk.CTkEntry(segment_frame, placeholder_text="Nhập transcript tại đây...")
            entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
            if 'transcript' in segment_data: entry.insert(0, segment_data['transcript'])
            
            delete_button = ctk.CTkButton(segment_frame, text="🗑️", width=30, fg_color="transparent", hover_color="#C82333", command=lambda data=segment_data: self.delete_segment(data))
            delete_button.grid(row=0, column=3, padx=5, pady=5)

            segment_data.update({'ui_frame': segment_frame, 'entry_widget': entry})
        
        self.save_metadata_button.configure(state="normal")

    def processing_finished(self, count):
        self.status_label.configure(text=f"Hoàn tất! Đã thêm {count} file mới cho '{self.current_speaker}'.")
        self.process_button.configure(state="normal")
        self.select_button.configure(state="normal")

    def play_audio_file(self, file_path):
        try:
            play(AudioSegment.from_file(file_path))
        except Exception as e:
            print(f"Lỗi khi phát file {os.path.basename(file_path)}: {e}")

    def delete_segment(self, segment_data):
        if messagebox.askyesno("Xác nhận Xóa", f"Bạn có chắc muốn xóa vĩnh viễn file:\n{os.path.basename(segment_data['path'])}?"):
            try:
                os.remove(segment_data['path'])
                segment_data['ui_frame'].destroy()
                self.audio_segments_data.remove(segment_data)
                self.status_label.configure(text=f"Đã xóa file {os.path.basename(segment_data['path'])}.")
            except Exception as e: messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")
            
    def save_metadata(self):
        if not self.current_speaker:
            messagebox.showerror("Lỗi", "Vui lòng chọn một người nói trước khi lưu.")
            return
            
        all_metadata = []
        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Giữ lại dữ liệu của những người nói khác
                    if not row['audio_filename'].startswith(self.current_speaker + "/"):
                        all_metadata.append(row)

        for segment_data in self.audio_segments_data:
            # [SỬA LỖI] Tạo đường dẫn tương đối theo đúng yêu cầu
            relative_path = os.path.join(self.current_speaker, os.path.basename(segment_data['path'])).replace("\\", "/")
            transcript = segment_data['entry_widget'].get().lower()
            if transcript:
                all_metadata.append({'audio_filename': relative_path, 'transcript': transcript})
        
        try:
            sorted_metadata = sorted(all_metadata, key=lambda x: x['audio_filename'])
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['audio_filename', 'transcript'])
                writer.writeheader()
                writer.writerows(sorted_metadata)
            messagebox.showinfo("Thành công", "Đã lưu tất cả thay đổi vào metadata.csv")
            self.status_label.configure(text="Đã lưu. Sẵn sàng cho tác vụ tiếp theo.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể ghi file CSV: {e}")

    def clear_speaker_directory(self):
        if not self.current_speaker:
            messagebox.showinfo("Thông báo", "Vui lòng chọn một người nói trước.")
            return
        
        speaker_path = os.path.join(self.wavs_dir_path, self.current_speaker)
        warning_message = f"CẢNH BÁO!\nHành động này sẽ XÓA TẤT CẢ các file .wav trong thư mục của người nói '{self.current_speaker}'.\n\nĐường dẫn: {speaker_path}\n\nHành động này không thể hoàn tác. Bạn có chắc chắn không?"
        
        if messagebox.askyesno("XÁC NHẬN XÓA", warning_message):
            files_deleted_count = 0
            try:
                for filename in os.listdir(speaker_path):
                    if filename.endswith('.wav'):
                        os.remove(os.path.join(speaker_path, filename))
                        files_deleted_count += 1
                self.load_speaker_session(self.current_speaker)
                messagebox.showinfo("Hoàn tất", f"Đã xóa thành công {files_deleted_count} file audio của {self.current_speaker}.")
                self.status_label.configure(text=f"Đã xóa sạch thư mục của {self.current_speaker}.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi dọn dẹp thư mục: {e}")
