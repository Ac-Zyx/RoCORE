# RoCORE
Codes for EMNLP2021 paper "A Relation-Oriented Clustering Method for Open Relation Extractiong".
https://aclanthology.org/2021.emnlp-main.765/
# Data
Download the dataset Tacred and FewRel and put it in data/tacred or data/fewrel.
# Usage
Train the model with:  
```
python main3.py --dataset ###(tacred or fewrel) --num_class ###(the number of predefined relation types) --new_class ###(the number of unknown relation types)
```
