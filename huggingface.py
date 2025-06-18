from pathlib import Path
from datasets import load_dataset, Audio, DatasetDict


AUDIO_FOLDER = "wavs"
METADATA_FILE = "metadata.csv"
HUB_REPO_ID = "hungkieu/vietnamese-voices"
# ==========================================================

def main():
    print(f"Đang tải metadata từ '{METADATA_FILE}'...")
    dataset = load_dataset("csv", data_files=METADATA_FILE, split="train")

    audio_folder_path = Path(AUDIO_FOLDER).resolve()

    print("Đang xử lý và liên kết các file audio trong các thư mục con...")
    
    def prepare_example(example):
        """Hàm này tìm đường dẫn đầy đủ đến file audio."""
        relative_path = example['audio_filename']
        example['audio'] = str(audio_folder_path / relative_path)
        return example

    dataset = dataset.map(prepare_example, num_proc=4)
    dataset = dataset.cast_column("audio", Audio())

    print("-> Xử lý hoàn tất. Dữ liệu đã sẵn sàng để tải lên.")
    print("\nVí dụ một bản ghi:")
    print(dataset[0])

    print(f"\nĐang tải dữ liệu lên repository: '{HUB_REPO_ID}'...")
    dataset.push_to_hub(repo_id=HUB_REPO_ID)

    print(f"\n🎉 Tải lên hoàn tất! Xem dataset tại: https://huggingface.co/datasets/{HUB_REPO_ID}")

if __name__ == "__main__":
    main()
