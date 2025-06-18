# Chứa các hằng số và giá trị cấu hình chung của ứng dụng
class AppConfig:
    # Tham số VAD mặc định
    VAD_THRESHOLD = 0.4
    VAD_MIN_SILENCE_MS = 500
    VAD_MIN_SPEECH_MS = 450
    VAD_SPEECH_PAD_MS = 300

    # Tham số Audio Export
    EXPORT_SAMPLE_RATE = 24000
    EXPORT_CHANNELS = 1
    
    # Các hằng số khác
    DEFAULT_PREFIX = "audio"
    WAVS_DIR = "wavs"
    METADATA_FILE = "metadata.csv"
