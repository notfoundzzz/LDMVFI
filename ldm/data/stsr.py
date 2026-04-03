import random
from functools import partial
from os import listdir
from os.path import join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import ldm.data.vfitransforms as vt


def _to_array(image):
    return np.array(image, dtype=np.float32)


def _normalize(frame):
    return frame / 127.5 - 1.0


def _downsample_bicubic(frame, scale_factor):
    h, w = frame.shape[:2]
    low_w = max(1, w // scale_factor)
    low_h = max(1, h // scale_factor)
    pil = Image.fromarray(frame.astype(np.uint8))
    return np.array(pil.resize((low_w, low_h), Image.BICUBIC), dtype=np.float32)


def _upsample_bicubic(frame, size):
    h, w = size
    pil = Image.fromarray(frame.astype(np.uint8))
    return np.array(pil.resize((w, h), Image.BICUBIC), dtype=np.float32)


class _TripletSuperResMixin:
    def __init__(self, scale_factor=2, upsample_lr=True):
        self.scale_factor = scale_factor
        self.upsample_lr = upsample_lr

    def _format_sample(self, raw_prev, raw_mid, raw_next):
        frame_prev = _to_array(raw_prev)
        frame_mid = _to_array(raw_mid)
        frame_next = _to_array(raw_next)

        prev_lr = _downsample_bicubic(frame_prev, self.scale_factor)
        next_lr = _downsample_bicubic(frame_next, self.scale_factor)

        if self.upsample_lr:
            prev_cond = _upsample_bicubic(prev_lr, frame_mid.shape[:2])
            next_cond = _upsample_bicubic(next_lr, frame_mid.shape[:2])
        else:
            prev_cond = frame_prev
            next_cond = frame_next

        return {
            "image": _normalize(frame_mid),
            "prev_frame": _normalize(prev_cond),
            "next_frame": _normalize(next_cond),
            "prev_frame_hr": _normalize(frame_prev),
            "next_frame_hr": _normalize(frame_next),
            "prev_frame_lr": _normalize(prev_lr),
            "next_frame_lr": _normalize(next_lr),
        }


class Vimeo90k_triplet_STSR(_TripletSuperResMixin, Dataset):
    def __init__(
        self,
        db_dir,
        train=True,
        crop_sz=(256, 256),
        augment_s=True,
        augment_t=True,
        scale_factor=2,
        upsample_lr=True,
    ):
        _TripletSuperResMixin.__init__(self, scale_factor=scale_factor, upsample_lr=upsample_lr)
        seq_dir = join(db_dir, "sequences")
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        seq_list_txt = join(db_dir, "sep_trainlist.txt" if train else "sep_testlist.txt")
        with open(seq_list_txt) as f:
            seq_path = [line.strip() for line in f.readlines() if line != "\n"]
        self.seq_path_list = [join(seq_dir, *line.split("/")) for line in seq_path]

    def __getitem__(self, index):
        raw_prev = Image.open(join(self.seq_path_list[index], "im3.png"))
        raw_mid = Image.open(join(self.seq_path_list[index], "im4.png"))
        raw_next = Image.open(join(self.seq_path_list[index], "im5.png"))

        if self.crop_sz is not None:
            raw_prev, raw_mid, raw_next = vt.rand_crop(raw_prev, raw_mid, raw_next, sz=self.crop_sz)
        if self.augment_s:
            raw_prev, raw_mid, raw_next = vt.rand_flip(raw_prev, raw_mid, raw_next, p=0.5)
        if self.augment_t:
            raw_prev, raw_mid, raw_next = vt.rand_reverse(raw_prev, raw_mid, raw_next, p=0.5)

        return self._format_sample(raw_prev, raw_mid, raw_next)

    def __len__(self):
        return len(self.seq_path_list)


class BVIDVC_triplet_STSR(_TripletSuperResMixin, Dataset):
    def __init__(
        self,
        db_dir,
        crop_sz=(256, 256),
        augment_s=True,
        augment_t=True,
        scale_factor=2,
        upsample_lr=True,
    ):
        _TripletSuperResMixin.__init__(self, scale_factor=scale_factor, upsample_lr=upsample_lr)
        db_dir = join(db_dir, "quintuplets")
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir)]

    def __getitem__(self, index):
        cat = Image.open(join(self.seq_path_list[index], "quintuplet.png"))
        raw_prev = cat.crop((256, 0, 256 * 2, 256))
        raw_next = cat.crop((256 * 2, 0, 256 * 3, 256))
        raw_mid = cat.crop((256 * 4, 0, 256 * 5, 256))

        if self.crop_sz is not None:
            raw_prev, raw_mid, raw_next = vt.rand_crop(raw_prev, raw_mid, raw_next, sz=self.crop_sz)
        if self.augment_s:
            raw_prev, raw_mid, raw_next = vt.rand_flip(raw_prev, raw_mid, raw_next, p=0.5)
        if self.augment_t:
            raw_prev, raw_mid, raw_next = vt.rand_reverse(raw_prev, raw_mid, raw_next, p=0.5)

        return self._format_sample(raw_prev, raw_mid, raw_next)

    def __len__(self):
        return len(self.seq_path_list)


class BVI_Vimeo_triplet_STSR(Dataset):
    def __init__(
        self,
        db_dir,
        crop_sz=(256, 256),
        p_datasets=None,
        iter=False,
        samples_per_epoch=1000,
        scale_factor=2,
        upsample_lr=True,
    ):
        vimeo = Vimeo90k_triplet_STSR(
            join(db_dir, "vimeo_septuplet"),
            train=True,
            crop_sz=crop_sz,
            scale_factor=scale_factor,
            upsample_lr=upsample_lr,
        )
        bvidvc = BVIDVC_triplet_STSR(
            join(db_dir, "bvidvc"),
            crop_sz=crop_sz,
            scale_factor=scale_factor,
            upsample_lr=upsample_lr,
        )

        self.datasets = [vimeo, bvidvc]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter
        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)
        self.samples_per_epoch = samples_per_epoch

        self.accum = [0]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i - 1].__getitem__(index - self.accum[i - 1])
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        return dataset.__getitem__(random.randint(0, len(dataset) - 1))

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        return self.samples_per_epoch


class Vimeo90k_triplet_STSR_val(Vimeo90k_triplet_STSR):
    def __init__(self, db_dir, scale_factor=2, upsample_lr=True):
        super().__init__(
            db_dir=db_dir,
            train=False,
            crop_sz=(256, 256),
            augment_s=False,
            augment_t=False,
            scale_factor=scale_factor,
            upsample_lr=upsample_lr,
        )
