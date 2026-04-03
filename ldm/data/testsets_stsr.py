import ast
import os
import time
from os.path import exists, join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite

import utility


def _downsample(frame, scale_factor):
    h, w = frame.shape[-2:]
    return F.interpolate(frame, size=(h // scale_factor, w // scale_factor), mode="bicubic", align_corners=False)


class TripletTestSetSTSR:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def _predict(self, model, sample_func, prev_hr, next_hr, gt, sampler_kwargs):
        prev_lr = _downsample(prev_hr, self.scale_factor)
        next_lr = _downsample(next_hr, self.scale_factor)

        if hasattr(model, "sample_stsr"):
            return model.sample_stsr(prev_lr, next_lr, gt.shape[-2:], **sampler_kwargs)

        prev_up = F.interpolate(prev_lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        next_up = F.interpolate(next_lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        xc = {"prev_frame": prev_up, "next_frame": next_up}
        c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
        shape = (model.channels, c.shape[2], c.shape[3])
        out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
        if isinstance(out, tuple):
            out = out[0]
        out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
        return torch.clamp(out, min=-1.0, max=1.0)

    def eval(self, model, sample_func, metrics=None, output_dir=None, output_name="output.png", resume=False, sampler_kwargs=None):
        metrics = metrics or ["PSNR", "SSIM", "LPIPS"]
        sampler_kwargs = sampler_kwargs or {}
        results_dict = {k: [] for k in metrics}
        start_idx = 0
        if resume:
            assert exists(join(output_dir, "results.txt")), "no res file found to resume from!"
            with open(join(output_dir, "results.txt"), "r") as f:
                for line in f.readlines():
                    if len(line) < 2:
                        continue
                    cur_res = ast.literal_eval(line.strip().split("-- ")[1].split("time")[0])
                    for k in metrics:
                        results_dict[k].append(float(cur_res[k]))
                    start_idx += 1

        logfile = open(join(output_dir, "results.txt"), "a")
        for idx in range(len(self.im_list)):
            if resume and idx < start_idx:
                assert exists(join(output_dir, self.im_list[idx], output_name))
                continue

            t0 = time.time()
            name = self.im_list[idx]
            print(f"Evaluating {name}")
            os.makedirs(join(output_dir, name), exist_ok=True)

            with torch.no_grad():
                with model.ema_scope() if hasattr(model, "ema_scope") else torch.no_grad():
                    out = self._predict(
                        model,
                        sample_func,
                        self.input0_list[idx],
                        self.input1_list[idx],
                        self.gt_list[idx],
                        sampler_kwargs,
                    )

            gt = self.gt_list[idx]
            prev_hr = self.input0_list[idx]
            next_hr = self.input1_list[idx]
            for metric in metrics:
                score = getattr(utility, f"calc_{metric.lower()}")(gt, out, [prev_hr, next_hr])[0].item()
                results_dict[metric].append(score)

            imwrite(out, join(output_dir, name, output_name), value_range=(-1, 1), normalize=True)
            msg = "{:<15s} -- {}".format(name, {k: round(results_dict[k][-1], 3) for k in metrics})
            msg += f"    time taken: {round(time.time()-t0, 2)}\n"
            print(msg, end="")
            logfile.write(msg)

        msg = "{:<15s} -- {}".format("Average", {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + "\n\n"
        print(msg, end="")
        logfile.write(msg)
        logfile.close()


class Middlebury_others_STSR(TripletTestSetSTSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(scale_factor=scale_factor)
        self.im_list = ["Beanbags", "Dimetrodon", "DogDance", "Grove2", "Grove3", "Hydrangea", "MiniCooper", "RubberWhale", "Urban2", "Urban3", "Venus", "Walking"]
        self.input0_list, self.input1_list, self.gt_list = [], [], []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(join(db_dir, "input", item, "frame10.png"))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, "input", item, "frame11.png"))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, "gt", item, "frame10i11.png"))).cuda().unsqueeze(0))


class Ucf_STSR(TripletTestSetSTSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(scale_factor=scale_factor)
        self.im_list = os.listdir(db_dir)
        self.input0_list, self.input1_list, self.gt_list = [], [], []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(join(db_dir, item, "frame_00.png"))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, item, "frame_02.png"))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, item, "frame_01_gt.png"))).cuda().unsqueeze(0))


class Snufilm_STSR(TripletTestSetSTSR):
    def __init__(self, db_dir, mode, scale_factor=2):
        super().__init__(scale_factor=scale_factor)
        self.mode = mode
        self.input0_list, self.input1_list, self.gt_list = [], [], []
        with open(join(db_dir, f"test-{mode}.txt"), "r") as f:
            triplet_list = f.read().splitlines()
        self.im_list = []
        for i, triplet in enumerate(triplet_list, 1):
            self.im_list.append(f"{mode}-{str(i).zfill(3)}")
            lst = triplet.split(" ")
            self.input0_list.append(self.transform(Image.open(join(db_dir, lst[0]))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, lst[2]))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, lst[1]))).cuda().unsqueeze(0))


class Snufilm_easy_STSR(Snufilm_STSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(db_dir[:-5], "easy", scale_factor=scale_factor)


class Snufilm_medium_STSR(Snufilm_STSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(db_dir[:-7], "medium", scale_factor=scale_factor)


class Snufilm_hard_STSR(Snufilm_STSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(db_dir[:-5], "hard", scale_factor=scale_factor)


class Snufilm_extreme_STSR(Snufilm_STSR):
    def __init__(self, db_dir, scale_factor=2):
        super().__init__(db_dir[:-8], "extreme", scale_factor=scale_factor)
