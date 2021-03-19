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
    model.compile(loss=masked_crossentropy, optimizer=Adam(3e-5))
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
    def __init__(self, model, data):
        self.best_val_score = 0.
        self.data = data
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(self.data, self.model)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.model.save_weights('best_model.h5')
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )