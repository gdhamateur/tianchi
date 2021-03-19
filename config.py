class BaseConfig:
    # config_path = '/data/home/gdh/external_model/keras/chinese_L-12_H-768_A-12/bert_config.json'
    # checkpoint_path = '/data/home/gdh/external_model/keras/chinese_L-12_H-768_A-12/bert_model.ckpt'
    # dict_path = '/data/home/gdh/external_model/keras/chinese_L-12_H-768_A-12/vocab.txt'
    config_path = '/data/home/gdh/external_model/keras/NEZHA-base/bert_config.json'
    checkpoint_path = '/data/home/gdh/external_model/keras/NEZHA-base/model.ckpt-900000'
    dict_path = '/data/home/gdh/external_model/keras/NEZHA-base/vocab.txt'
    train_path = '/data/home/gdh/dataset/天池-语义匹配/train.tsv'
    test_path = '/data/home/gdh/dataset/天池-语义匹配/testA.tsv'
    min_count = 5
    max_len = 32
    batch_size = 128