from os.path import join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Vimeo90kTripletSTSRLoRA(Dataset):
    def __init__(
        self,
        db_dir_hr,
        db_dir_lr,
        list_file,
        split=None,
        max_samples=0,
    ):
        self.db_dir_hr = db_dir_hr
        self.db_dir_lr = join(db_dir_lr, split) if split else db_dir_lr
        self.seq_dir_hr = join(self.db_dir_hr)
        self.seq_dir_lr = join(self.db_dir_lr)

        with open(list_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if max_samples > 0:
            lines = lines[:max_samples]
        self.sample_ids = lines

    def __len__(self):
        return len(self.sample_ids)

    def _read(self, path):
        frame = np.array(Image.open(path), dtype=np.float32)
        return frame / 127.5 - 1.0

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        hr_dir = join(self.seq_dir_hr, *sample_id.split("/"))
        lr_dir = join(self.seq_dir_lr, *sample_id.split("/"))

        image = self._read(join(hr_dir, "im4.png"))
        prev_lr = self._read(join(lr_dir, "im3.png"))
        next_lr = self._read(join(lr_dir, "im5.png"))
        lr_sequence = np.stack(
            [self._read(join(lr_dir, f"im{i}.png")) for i in range(1, 7)],
            axis=0,
        )

        return {
            "image": image,
            "prev_frame_lr": prev_lr,
            "next_frame_lr": next_lr,
            "lr_sequence": lr_sequence,
            "sample_id": sample_id,
        }
