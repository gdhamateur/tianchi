#! -*- coding:utf-8 -*-
# 句子对分类任务，脱敏数据
# 比赛链接：https://tianchi.aliyun.com/competition/entrance/531851

import os
from config import BaseConfig
from util import load_data, get_tokens, get_keep_tokens, adversarial_training
from model import get_model, Evaluator
from dataset import data_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载数据集
data = load_data(
    BaseConfig.train_path
)
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data(
    BaseConfig.test_path
)

# 数据集词频
tokens = get_tokens(data+test_data)

# BERT词频
keep_tokens = get_keep_tokens()

# 模拟未标注
for d in valid_data + test_data:
    train_data.append((d[0], d[1], -5))
#
# 加载预训练模型
model = get_model(tokens, keep_tokens)
#
# 转换数据集
train_generator = data_generator(train_data, BaseConfig.batch_size)
valid_generator = data_generator(valid_data, BaseConfig.batch_size)


if __name__ == '__main__':

    evaluator = Evaluator(model, valid_generator)
    adv_layer_names = ['Embedding-Token']
    adversarial_training(model, adv_layer_names, 0.5)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=100,
        callbacks=[evaluator]
    )
