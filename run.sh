source conda create --name jinmang2 python=3.7
source activate jinmang2
pip install datasets transformers tokenizers
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
python run.py
