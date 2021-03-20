import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, K
from bert4keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from config import BaseConfig


# mask掉非预测部分
def masked_crossentropy(y_true, y_pred):
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(y_true, 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]


def get_model(tokens, keep_tokens):
    model = build_transformer_model(
        config_path=BaseConfig.config_path,
        checkpoint_path=BaseConfig.checkpoint_path,
        with_mlm=True,
        model="nezha",
        keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)]
    )
    model.compile(loss=masked_crossentropy, optimizer=Adam(2e-5))
    model.summary()
    return model


def evaluate(data, model):
    """线下评测函数
    """
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        y_true = y_true[:, 0] - 5
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
    return roc_auc_score(Y_true, Y_pred)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model, data, fold_num, is_warmup):
        self.best_val_score = 0.
        self.data = data
        self.model = model
        self.fold_num = fold_num
        self.is_warmup = is_warmup

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            if self.is_warmup:
                self.model.save_weights('checkout/warmup/model-{}-{}.h5'.format(epoch, self.fold_num))
            else:
                self.model.save_weights('checkout/no_warmup/model-{}-{}.h5'.format(epoch, self.fold_num))
        val_score = evaluate(self.data, self.model)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            if self.is_warmup:
                self.model.save_weights('checkout/warmup/best_model/best_model_{}.h5'.format(self.fold_num))
            else:
                self.model.save_weights('checkout/no_warmup/best_model/best_model_{}.h5'.format(self.fold_num))
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )


class Warmup(keras.callbacks.Callback):
    def __init__(self, lr_base=3e-5, lr_min=0.0, decay=0, warmup_epochs=2):
        self.num_passed_batchs = 0
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base
        self.lr_min = lr_min
        self.decay = decay
        self.steps_per_epoch = 0

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch == 0:
            # 防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr * ((1 - self.decay) ** (
                                    self.num_passed_batchs - self.steps_per_epoch * self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
        print("learn-rate: {}".format(K.get_value(self.model.optimizer.lr)))