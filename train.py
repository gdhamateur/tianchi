#! -*- coding:utf-8 -*-
# 句子对分类任务，脱敏数据
# 比赛链接：https://tianchi.aliyun.com/competition/entrance/531851

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import K
from config import BaseConfig
from util import load_data, get_tokens, get_keep_tokens, adversarial_training
from model import get_model, Evaluator, Warmup
from dataset import data_generator
from random import seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 2021

if not os.path.exists("checkout"):
    os.makedirs("checkout")
    os.makedirs("checkout/warmup")
    os.makedirs("checkout/no_warmup")
    os.makedirs("checkout/warmup/best_model")
    os.makedirs("checkout/no_warmup/best_model")

# 加载数据集
data = load_data(
    BaseConfig.train_path
)
test_data = load_data(
    BaseConfig.test_path
)

# 数据集词频
tokens = get_tokens(data+test_data)

# BERT词频
keep_tokens = get_keep_tokens()

# # 模拟未标注
# for d in test_data:
#     data.append((d[0], d[1], -5))
labels = []
for d in data:
    labels.append(d[2])


if __name__ == '__main__':
    skf = StratifiedKFold(5, shuffle=True, random_state=2021)
    for fold_id, (train_id, valid_id) in enumerate(skf.split(range(len(data)), labels)):
        train_data = np.array(data)[train_id]
        valid_data = np.array(data)[valid_id]
        train_data = list(train_data)
        valid_data = list(valid_data)
        for d in test_data:
            train_data.append((d[0], d[1], -5))
        train_generator = data_generator(train_data, BaseConfig.batch_size)
        valid_generator = data_generator(valid_data, BaseConfig.batch_size)
        K.clear_session()
        seed(SEED + fold_id)
        np.random.seed(SEED + fold_id)
        tf.random.set_random_seed(SEED + fold_id)
        # 加载预训练模型
        model = get_model(tokens, keep_tokens)
        adv_layer_names = ['Embedding-Token']
        adversarial_training(model, adv_layer_names, 0.5)
        evaluator = Evaluator(model, valid_generator, fold_id, True)
        warmup = Warmup(decay=2e-5, warmup_epochs=3)
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=80,
            callbacks=[evaluator, warmup]
        )
        del train_data, valid_data, model

