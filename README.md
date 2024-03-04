```
conda create --name onestop python==3.9.18
conda activate onestop

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

bash downloadModels.sh

steamlit run app.py
```