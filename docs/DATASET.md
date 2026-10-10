# Dataset and training guide

The benchmark uses Northeastern University's [NEU surface defect classification dataset](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) (NEU-CLS): 1,800 grayscale images, six classes, 300 images per class. NEU-DET is a different detection dataset and is not used here.

Run commands from the repository root after installing the development dependencies in `.venv`. The examples use Windows PowerShell; replace the interpreter with `.venv/bin/python` on Linux/macOS.

## Download and organize

```powershell
& .\.venv\Scripts\python.exe -m src.download_data
```

The downloader retrieves the official archive and organizes original images into `data/raw/{class_name}/`. RAR extraction requires a compatible `tar` implementation. If automatic download or extraction fails, download NEU-CLS through the official dataset page and use either a local archive or an extracted image directory:

```powershell
& .\.venv\Scripts\python.exe -m src.download_data --archive "C:\path\to\NEU-CLS.rar"
& .\.venv\Scripts\python.exe -m src.download_data --extracted-dir "C:\path\to\NEU-CLS"
```

Choose one source. The extracted directory should contain the original BMP images (`Cr_1.bmp`, `In_1.bmp`, and so on); the organizer searches recursively. Source metadata is written to `data/raw/dataset_source.json`. The dataset is not included in Git; follow its provider's usage and attribution terms.

## Deterministic splits

```powershell
& .\.venv\Scripts\python.exe -m src.split_data --seed 42
& .\.venv\Scripts\python.exe -m src.split_data --verify-only
```

Outputs are `data/processed/{train,val,test}/{class_name}/` with 1,260 training, 270 validation and 270 test images. Each class contributes 210/45/45 images. Reruns verify and reuse a matching managed split; inconsistent or unmanaged output is rejected. Avoid rebuilding splits used for an existing evaluation.

Images remain unchanged on disk. DataLoaders apply resize to 224 × 224, grayscale-to-three-channel conversion and ImageNet normalization in memory. Training adds horizontal/vertical flips and rotations up to 10 degrees; validation/test use no augmentation.

## Reproduce training

```powershell
& .\.venv\Scripts\python.exe -m src.train --architecture custom_cnn
& .\.venv\Scripts\python.exe -m src.train --architecture resnet18
```

CUDA is selected when available. Add `--require-cuda` to require a GPU; use `--batch-size` to choose the batch size. ResNet18 downloads official TorchVision ImageNet weights for training, trains the classifier with a frozen backbone, then fine-tunes `layer4`. Training uses seed 42, AdamW, cross-entropy and early stopping, selecting checkpoints by validation macro-F1.

Training writes `artifacts/models/baseline_cnn.pth` or `artifacts/models/resnet18_best.pth` and reports under `artifacts/reports/`. Back up existing checkpoints/reports before running training or evaluation. Reproduction can differ across hardware and library versions; the README metrics describe the original release run.

## Evaluate and explain

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate --audit-only
& .\.venv\Scripts\python.exe -m src.evaluate
& .\.venv\Scripts\python.exe -m src.explain --samples-per-class 1
```

Evaluation audits readability, split counts and cross-split exact/near duplicates before using the test split. Potential leakage blocks test inference. A clean audit enables classification metrics, confusion matrices and misclassification reports for both models. Keep the test set out of training and hyperparameter selection.

Grad-CAM demonstrations use validation images and the saved ResNet18 checkpoint. The command-line explanation script requires CUDA; CPU explanations are available through `InferencePipeline(device="cpu").explain(...)` and the dashboard. Grad-CAM runs with gradients and highlights class evidence rather than segmenting defect boundaries.

To deploy a reproduced model, replace the deployment manifest with its verified size/hash and supply evaluation results for those weights. Do not present the release benchmark as the performance of a replacement checkpoint.
