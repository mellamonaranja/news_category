import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
from transformers import AutoTokenizer

from dataloader import load
from metrics import custom_confusion_matrix
from models.MainModels import EncoderModel


@hydra.main(config_name="config.yml")
def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    clf = None
    try:
        clf = EncoderModel.load(cfg.ETC.output_dir2)
    except Exception as e:
        print(e)
        print("Cannot load model")
        print(cfg)

    print(clf.get_config())

    # y = [d["labels"] for d in eval_dataset]
    tf_data = {k: tf.constant(v) for k, v in eval_dataset[:].items()}

    y = tf_data["labels"]
    pred = clf(**tf_data)  # tensor로 바꿔줘야함

    result = custom_confusion_matrix(y, pred)
    print(result)


if __name__ == "__main__":
    main()
