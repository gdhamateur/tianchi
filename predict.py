from tqdm import tqdm
from util import load_data, get_tokens, get_keep_tokens
from config import BaseConfig
from model import get_model
from dataset import data_generator


# 加载数据集
data = load_data(
    BaseConfig.train_path
)
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data(
    BaseConfig.test_path
)

test_generator = data_generator(test_data, BaseConfig.batch_size)
# 数据集词频
tokens = get_tokens(data+test_data)

# BERT词频
keep_tokens = get_keep_tokens()

model = get_model(tokens, keep_tokens)
model.load_weights('best_model.h5')

F = open("result.csv", mode="w")
for x_true, _ in tqdm(test_generator):
    y_pred = model.predict(x_true)[:, 0, 5:7]
    y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
    for p in y_pred:
        F.write('%f\n' % p)
F.close()