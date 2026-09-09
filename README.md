# Pokemon TCG AI: Neural Architecture Comparison for Move Prediction

**Authors:** Nicole Zanello & Philipp Zengl  
**Institution:** Technical University of Munich - Heilbronn Campus  
**Course:** Deep Learning and Decision Making (Summer 2026)

## Project Overview

This project systematically compares three neural network architectures (MLP, LSTM, Transformer) for predicting optimal moves in Pokemon Trading Card Game battles. Using 5,198 competitive games with 250,000+ decision points, we achieved 70.83% test accuracy with an LSTM model featuring focal loss optimization.

**Research Question:** Which neural architecture best learns sequential decision-making from limited expert gameplay data?

## Key Results

- **Best Model:** LSTM (768×3 layers) + Focal Loss (γ=2.0)
- **Test Accuracy:** 70.83% (Top-3: 88.43%)
- **Key Finding:** LSTMs outperform Transformers on limited-data sequential tasks
- **Dataset:** 5,198 competitive games, 250,000+ decision points, 24 trained model variants

## Repository Structure

```
.
├── README.md                 # This file
├── report_final.pdf          # Final academic report
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── feature_extractor.py # Feature engineering
│   ├── build_dataset.py     # Dataset construction
│   ├── train_MLP.py         # MLP baseline training
│   ├── train_MLP_improved.py
│   ├── train_LSTM.py        # LSTM baseline training
│   ├── train_LSTM_improved.py
│   ├── train_Transformer.py # Transformer baseline
│   ├── train_Transformer_improved.py
│   ├── train_ensemble.py    # Ensemble methods
│   ├── ensemble_evaluation.py
│   └── compare_models.py    # Model comparison
└── models/                   # Trained model checkpoints
    ├── MLP_baseline.pt
    ├── LSTM_baseline.pt
    ├── Transformer_baseline.pt
    └── lstm_best_70.83acc.pt
```

## Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- 16GB RAM minimum

### Installation

```bash
git clone https://github.com/555p9psh8c-eng/DLDM_Pokemon_Project.git
cd DLDM_Pokemon_Project
pip install -r requirements.txt
```

### Download Dataset

**Dataset Sources (both contain the same episode data):**
- Kaggle Dataset: https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-08
- Kaggle Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

**Building the Dataset:**

After downloading the raw episode data from Kaggle:

```bash
# Extract episodes archive to project directory
# The dataset builder will process raw game logs into training data

python src/build_dataset.py \
    --episodes_dir episodes/ \
    --output_path dataset.pt
```

This creates `dataset.pt` with extracted features and labels from all games.

**Full Model Archive** (optional - all 24 trained models):
- SharePoint: https://sap-my.sharepoint.com/:f:/p/philipp_zengl/IgCnz8TTgFmDTpADWEWO7qqtAUiBCvuE_KJUQxx078ZRMRU
- Note: Repository includes baseline + best models. Full archive has all experimental variants.
- **Access Issues?** Contact philipp.zengl@tum.de or nicole.zanello@tum.de

### Training from Scratch

```bash
# Train the best LSTM model
python src/train_LSTM_improved.py \
    --seed 456 \
    --lstm_hidden 768 \
    --lstm_layers 3 \
    --focal_gamma 2.0 \
    --epochs 30 \
    --batch_size 256 \
    --dataset_path dataset.pt
```

Expected training time: 12-15 hours on NVIDIA RTX 3090

### Evaluate Pre-trained Models

```bash
python src/compare_models.py --model_dir models/ --dataset_path dataset.pt
```

## Results Summary

### Architecture Comparison

| Architecture | Test Accuracy | Top-3 Accuracy | Parameters |
|--------------|---------------|----------------|------------|
| MLP          | 54.75%        | 80.54%         | 1.35M      |
| LSTM         | 65.41%        | 84.86%         | 6.5M       |
| Transformer  | 66.26%        | 86.49%         | 2.8M       |

### Best Model Performance

| Model Configuration | Test Accuracy | Parameters |
|---------------------|---------------|------------|
| LSTM 768×3 + Focal Loss | **70.83%** | 13.8M |

See [report_final.pdf](report_final.pdf) for complete results and analysis.

## Project Deliverables

- 📄 **Final Report:** [report_final.pdf](report_final.pdf) - Academic report with full analysis
- 🎥 **Video Presentation:** [YouTube](https://www.youtube.com/watch?v=B7njk6b17tI)
- 💾 **Trained Models:** Included in `models/` + [Full Archive (SharePoint)](https://sap-my.sharepoint.com/:f:/p/philipp_zengl/IgCnz8TTgFmDTpADWEWO7qqtAUiBCvuE_KJUQxx078ZRMRU)
- 💻 **Source Code:** Training scripts in `src/`

## Reproducibility

### System Requirements
- Python 3.9+ (tested on 3.10)
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB RAM (32GB recommended)
- ~10GB disk space (excluding dataset)

### Step-by-Step Reproduction

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download and Build Dataset**
   - Download raw episodes from Kaggle (links above)
   - Extract to `episodes/` directory
   - Run dataset builder:
     ```bash
     python src/build_dataset.py --episodes_dir episodes/ --output_path dataset.pt
     ```

3. **Train Models**
   ```bash
   # Baseline models
   python src/train_MLP.py --seed 0 --dataset_path dataset.pt
   python src/train_LSTM.py --seed 0 --dataset_path dataset.pt
   python src/train_Transformer.py --seed 0 --dataset_path dataset.pt
   
   # Best model
   python src/train_LSTM_improved.py \
       --seed 456 \
       --lstm_hidden 768 \
       --lstm_layers 3 \
       --focal_gamma 2.0 \
       --dataset_path dataset.pt
   ```

4. **Compare Results**
   ```bash
   python src/compare_models.py --model_dir models/ --dataset_path dataset.pt
   ```

Full methodology and hyperparameters available in the report.

## Citation

```bibtex
@article{zanello2026pokemon,
  title={Comparing Neural Network Architectures for Pokemon TCG Move Prediction},
  author={Zanello, Nicole and Zengl, Philipp},
  institution={Technical University of Munich - Heilbronn Campus},
  year={2026}
}
```

## Authors

- **Nicole Zanello** (Matriculation: 03826806)
- **Philipp Zengl** (Matriculation: 03827436)
- **Contact:** nicole.zanello@tum.de, philipp.zengl@tum.de

## License

This project is developed for academic purposes as part of the Deep Learning and Decision Making course at TUM.

---

**Technical University of Munich - Heilbronn Campus**  
**Deep Learning and Decision Making | Summer 2026**
