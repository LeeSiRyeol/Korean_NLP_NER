# Korean_NLP_NER

## Enviroments
-[Ubuntu 20.04]

-[Python>=3.8]

## Installation
Please go ahead and install the Python package to run.

```sh
$pip3 install -r requirements.txt
```

## Korea NLP models
|Model|url||
|----------|-----------|-----------|
|KLUE-BERT|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KLUE-RoBERTa|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KoBERT|https://github.com/SKTBrain/KoBERT.git|-|
|KorBERT|https://aiopen.etri.re.kr/bertModel|KorBERT can be downloaded and used with the permission of ETRI|
|KoBigBird|https://github.com/monologg/KoBigBird.git|Use and Support on Hugging Face|

* KoBigBird can handle more than 512 tokens, with a maximum of 4096 tokens.
* If you are using KorBERT, you need to download the model through ETRI and then store KorBERT in the 'module' folder. Among the models supported by ETRI, we used kor_tensorflow. Upload the required files in a blank state.

## Model Train
Each NLP model can be trained using the `train.py` script.
```sh
$python3 train.py
```
