# machine-translation
Russian to English machine transformer using Transformer.

### Preparation:
Unpack models archieve in root folder: https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz

Make sure dependencies from https://github.com/pytorch/fairseq/tree/master/examples/wmt19 satisfyed:
```
pip install fastBPE sacremoses
```


### Run:
```
python nmt.py
```

it will translate ```test.ru.txt``` file into ```answer.txt```
