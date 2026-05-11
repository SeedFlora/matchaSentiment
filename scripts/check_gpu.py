from __future__ import annotations

import json

import torch


def main() -> None:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
