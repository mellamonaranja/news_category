import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_addons as tfa

# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix


class CustomF1Score(tfa.metrics.F1Score):
    __name__ = "CustomF1Score"

    def __init__(self, num_classes, average=None, threshold=0.5, **kwargs) -> None:
        self.num_classes = num_classes
        super().__init__(
            self.num_classes, average=average, threshold=threshold, **kwargs
        )

    def __call__(self, y, pred):
        pred = tf.convert_to_tensor(pred)
        y = tf.cast(y, pred.dtype)
        result = super().__call__(y, pred)
        return result


# def custom_confusion_matrix(y, pred):
#     # y = tf.reshape(
#     #     y,
#     #     [-1],
#     # )
#     # pred = tf.reshape(
#     #     pred,
#     #     [-1],
#     # )
#     print(type(y.numpy()))
#     print(type(pred.numpy()))
#     # pred = tf.convert_to_tensor(pred)
#     # y = tf.cast(y, pred.dtype)
#     result = tf.math.confusion_matrix(y, pred, num_classes=30)

#     return result


# def custom_confusion_matrix(cfm, classes):
#     # cfm = tf.reshape(
#     #     y,
#     #     [-1],
#     # )
#     # classes = tf.reshape(
#     #     pred,
#     #     [-1],
#     # )
#     # print(y)
#     # print(pred)

#     # cfm = [[35, 0, 6], [0, 0, 3], [5, 50, 1]]
#     # classes = ["0", "1", "2"]

#     df_cfm = pd.DataFrame(cfm.numpy(), columns=classes.numpy())
#     print(df_cfm)
#     plt.figure(figsize=(10, 7))
#     cfm_plot = sn.heatmap(df_cfm, annot=True)
#     cfm_plot.figure.savefig("cfm.png")


loss = tf.keras.losses.binary_crossentropy
accuracy = tf.keras.metrics.binary_accuracy
