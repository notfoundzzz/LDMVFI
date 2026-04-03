import importlib
import sys

MODULES = [
    "torch",
    "torchvision",
    "pytorch_lightning",
    "omegaconf",
    "einops",
    "cv2",
    "imageio",
    "timm",
    "cupy",
]


def main():
    print("python", sys.version.replace("\n", " "))
    failed = False
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(name, "OK", getattr(mod, "__version__", "nover"))
        except Exception as exc:
            failed = True
            print(name, "FAIL", type(exc).__name__, str(exc))

    try:
        import torch

        print("cuda_available", torch.cuda.is_available())
        print("device_count", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("device0", torch.cuda.get_device_name(0))
    except Exception as exc:
        failed = True
        print("torch_cuda_check", "FAIL", type(exc).__name__, str(exc))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
