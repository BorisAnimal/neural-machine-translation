import torch


# Models archieve: https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz
# 
# Make sure dependencies from https://github.com/pytorch/fairseq/tree/master/examples/wmt19 satisfyed:
# ```
# pip install fastBPE sacremoses
# ```


inp_path = "test.ru.txt"
out_path = "answer.txt"
with open(inp_path, 'r', encoding='utf-8') as f:
    rus = f.readlines()


# Russian to English translation
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
en = [ru2en.translate(sent) for sent in rus]

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("".join(en))

