import pandas as pd 
import tensorflow as tf 
import numpy as np
import transformers
import matplotlib.pyplot as plt 

from transformers import BertTokenizer

# load pre-trained tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

# Text 
text = 'ini adalah contoh puisi yang akan di proses kedepannya. Nanti text nya bisa dimodif atau diubah kok, yang penting bentuk text nya seperti ini.'
# tokenize text
input_ids = tokenizer.encode(text, padding='max_length', max_length=512)

# print token ids 
print(input_ids)