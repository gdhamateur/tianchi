import numpy as np
from bert4keras.snippets import truncate_sequences
from bert4keras.tokenizers import load_vocab
from bert4keras.backend import keras, K
import json
from config import BaseConfig


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(BaseConfig.max_len, -1, a, b)
            D.append((a, b, c))
    return D


# 统计数据集词频
def get_tokens(data):
    # 统计词频
    tokens = {}
    for d in data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    tokens = {i: j for i, j in tokens.items() if j >= BaseConfig.min_count}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    # tokens = {词: 频率排名}
    tokens = {
        t[0]: i + 7
        for i, t in enumerate(tokens)
    }  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    return tokens


# 统计BERT词汇表词频
def get_keep_tokens():
    counts = json.load(open('counts.json'))
    del counts['[CLS]']
    del counts['[SEP]']
    token_dict = load_vocab(BaseConfig.dict_path)
    freqs = [
        counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    keep_tokens = list(np.argsort(freqs)[::-1])
    return keep_tokens


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def adversarial_training(model, embedding_names, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_names
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    embedding_layers = []
    for embedding_name in embedding_names:
        for output in model.outputs:
            embedding_layer = search_layer(output, embedding_name)
            if embedding_layer is not None:
                embedding_layers.append(embedding_layer)
                break
    for embedding_layer in embedding_layers:
        if embedding_layer is None:
            raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = [embedding_layer.embeddings for embedding_layer in embedding_layers]  # Embedding矩阵
    gradients = K.gradients(model.total_loss, embeddings)  # Embedding梯度
    # gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor
    gradients = [K.zeros_like(embedding) + gradient for embedding, gradient in zip(embeddings, gradients)]

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=gradients,
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        #         grads = embedding_gradients(inputs)[0]  # Embedding梯度
        #         delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        grads = embedding_gradients(inputs)  # Embedding梯度
        deltas = [epsilon * grad / (np.sqrt((grad ** 2).sum()) + 1e-8) for grad in grads]  # 计算扰动
        # 注入扰动
        # K.set_value(embeddings, K.eval(embeddings) + delta)
        for embedding, delta in zip(embeddings, deltas):
            K.set_value(embedding, K.eval(embedding) + delta)

        outputs = old_train_function(inputs)  # 梯度下降
        # 删除扰动
        # K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        for embedding, delta in zip(embeddings, deltas):
            K.set_value(embedding, K.eval(embedding) - delta)
        return outputs

    model.train_function = train_function  # 覆盖原训练函数