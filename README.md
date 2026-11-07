# Brain Tumor Detection & Classification with Deep Learning

An educational Gradio demo for brain MRI classification with **MobileNetV2** and **Grad-CAM**.

## Launch the app

After the first installation below, run:

```bash
python app.py
```

Open the local URL printed in the terminal, normally **http://127.0.0.1:7860**. Upload an anonymized image and click **Analyser l’image**.

The interface is designed for a **PC browser**, with the MRI and Grad-CAM shown side by side. A window width of at least 1280 pixels is recommended.

The first launch downloads about **42 MB** of pretrained models from this repository’s release and verifies their SHA-256 hashes. Later launches reuse the local files. **No training, dataset download or GPU is required to try the demo.** An internet connection is needed for the first model download.

<details>
<summary><strong>First installation — Python 3.12</strong></summary>

```bash
git clone https://github.com/Irfat69/Projet_Tumeur.git
cd Projet_Tumeur
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux/WSL commands are shown. On Windows, activate the environment with `venv\Scripts\activate`.

</details>

![Actual Gradio application](https://github.com/Irfat69/Projet_Tumeur/releases/download/gradio-demo-v3/gradio_demo_desktop.png)

This authentic example is annotated **meningioma** and predicted **glioma**; it was selected without searching for a successful prediction. The image derives from Jun Cheng’s [Figshare dataset](https://doi.org/10.6084/m9.figshare.1512427), retrieved through the pinned Kaggle mirror. The screenshot is shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Scope and scientific limits

The app predicts tumor/no tumor and, conditionally, glioma/meningioma/pituitary. **It is a research demonstration, not a diagnostic tool.** Scores are uncalibrated, Grad-CAM is not a validated tumor localization, and arbitrary non-MRI inputs are not reliably rejected.

The corrected models obtained **97.26% binary accuracy** and **83.25% three-class accuracy** on exploratory internal tests. On a differently prepared, positive-only BraTS cohort of 336 subjects, binary recall fell to **48.81%** and glioma-class recall to **2.98%**. Unknown JPEG subject identities and earlier test reuse also limit interpretation. These are not clinically validated performances.

[Pretrained models, model card and downloadable scientific audit](https://github.com/Irfat69/Projet_Tumeur/releases/tag/gradio-demo-v3).

<details>
<summary><strong>Training and development</strong></summary>

The reusable pipeline is in `src/brain_tumor/`. To prepare the pinned datasets and train new runs:

```bash
python -m brain_tumor.data_audit --download
python train.py --task binary --run-dir outputs/runs/my_binary
python train.py --task multiclass --run-dir outputs/runs/my_multiclass
```

All run artifacts stay local. To use your new models:

```bash
MRI_BINARY_RUN=outputs/runs/my_binary \
MRI_MULTICLASS_RUN=outputs/runs/my_multiclass \
python app.py
```

For Linux/WSL NVIDIA support, install `requirements/gpu.txt` and run `python -m brain_tumor.gpu`. Tests run on CPU without downloading models or datasets:

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

The scientific archive records source versions, patient/export grouping, conflicts, baselines, uncertainty and external failure. Data sources are SARTAJ, Nickparvar, Praneet and Cheng/Figshare; they overlap and are not four independent cohorts. Comply with the original data terms. No blanket code/data license is granted by this repository.

</details>

**Irfat Fejzullahu** — Master 2 Data Science, Université Claude Bernard Lyon 1.
