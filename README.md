# bob
Bob knows his shit


python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# To train
Edit algebra.json
python train.py

# Export trained model to gguf
Keep algebra.json the same
python export-gguf.py 

# To run, just use normal llama.cpp