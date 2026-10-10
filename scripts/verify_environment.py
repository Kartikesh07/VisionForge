"""Verify Step 1 dependencies and GPU operation without datasets or models."""

import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "matplotlib": "matplotlib",
    "streamlit": "streamlit",
    "grad-cam": "pytorch_grad_cam",
    "tqdm": "tqdm",
    "pytest": "pytest",
}


def main():
    root = Path(__file__).resolve().parents[1]
    expected_prefix = root / ".venv"
    if Path(sys.prefix).resolve() != expected_prefix.resolve() or sys.prefix == sys.base_prefix:
        raise SystemExit("Run with this project's .venv Python interpreter.")

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "isolated_from_system_packages": not any(
            "include-system-site-packages = true" in line.lower()
            for line in (expected_prefix / "pyvenv.cfg").read_text().splitlines()
        ),
        "packages": {},
        "failures": [],
    }
    for distribution, module in DEPENDENCIES.items():
        try:
            imported = importlib.import_module(module)
            module_path = Path(imported.__file__).resolve()
            if not module_path.is_relative_to(expected_prefix.resolve()):
                raise RuntimeError(f"Package imported outside .venv: {module_path}")
            report["packages"][distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            report["failures"].append(f"{distribution}: {type(exc).__name__}: {exc}")

    if "torch" in report["packages"]:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["pytorch_cuda_runtime"] = torch.version.cuda
        report["gpu_count"] = torch.cuda.device_count()
        try:
            assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8
            report["cpu_tensor_check"] = "passed"
            if report["cuda_available"]:
                report["gpu_name"] = torch.cuda.get_device_name(0)
                tensor = torch.ones(2, 2, device="cuda")
                assert (tensor @ tensor).sum().item() == 8
                torch.cuda.synchronize()
                report["cuda_tensor_check"] = "passed"
            else:
                report["cuda_tensor_check"] = "unavailable"
        except Exception as exc:
            report["failures"].append(f"Tensor verification: {type(exc).__name__}: {exc}")

    output = root / "artifacts" / "reports" / "setup-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
