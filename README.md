# AI-based Emotion Recognition for Tailored Advertisements

Multimodal (vision + audio + text) emotion recognition with late fusion. Includes a dummy dataset to run end-to-end without external downloads.
It is also going for publication as Research Paper.


## Project Structure
```
AI_Emotion_Recognition/
├── data/
│   └── dummy_dataset.py
├── models/
│   ├── audio_model.py
│   ├── fusion_model.py
│   ├── text_model.py
│   └── vision_model.py
├── utils/
│   ├── evaluation.py
│   └── preprocess.py
├── train.py
├── evaluate.py
├── requirements.txt
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Train on dummy data (3 epochs)
python train.py

# Evaluate (loads checkpoint if present)
python evaluate.py
```
