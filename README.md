# Transfer-Learning-CNN-Tensorflow

This repository contains a TensorFlow-based implementation of transfer learning using EfficientNetV2 for a Food Vision dataset (with 10 classes). The primary focus is on fine-tuning EfficientNetV2B0 for image classification, utilizing callbacks to prevent overfitting and exponential learning rate scheduling for optimization.

## Dataset
The dataset used in this project is a subset of the [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), but instead of 101 labels, only 10 labels have been used. Each image corresponds to one of the food categories, and the dataset is split into training and testing sets.

- **Training Set**: Contains labeled images for model training.
- **Validation Set**: A separate portion of the data for evaluating model performance during training.


## Preprocessing
Images are preprocessed into a consistent shape and format using TensorFlow's `image_dataset_from_directory`, where images are automatically resized to the model's expected input shape (224x224) and batched for efficient processing.


```Python
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=_TRAINING_SET,
    image_size=_IMG_SIZE,
    label_mode="categorical",
    labels="inferred",
    batch_size=_BATCH_SIZE,

)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=_VALIDATION_SET,
    image_size=_IMG_SIZE,
    label_mode="categorical",
    labels="inferred",
    batch_size=_BATCH_SIZE,

)
```

# Training

## Transfer Learning With EfficientNetV2
The EfficientNetV2B0 model is used as the base model for transfer learning. The pre-trained weights on ImageNet are used, with the top classification layers removed. A global average pooling layer and a dense layer with softmax activation are added to predict the 10 food categories. The base model is initially frozen, and only the newly added layers are trained.

```Python
def train_model_0() -> tf.keras.Model:

    EfficientNetV2B0_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False
        )
    EfficientNetV2B0_model.trainable=False
    input_layer = tf.keras.layers.Input(
    shape=(224,224,3),
    name="Input_Layer"
    )
    x = EfficientNetV2B0_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="GAP2D_1"
    )(x)

    output_layer = tf.keras.layers.Dense(
        units=_OUTPUT_SIZE,
        activation="softmax",
        name="Ouput_Layer",
        kernel_regularizer=tf.keras.regularizers.L1()
    )(x)

    model_0 = tf.keras.Model(
        input_layer,
        output_layer,

        )

    return model_0
```

## Overfit Callback
The custom callback `TrainingCheckPoint` monitors the training process for signs of overfitting. It compares validation loss with training loss and stops training if overfitting persists beyond a defined patience.

```Python
class TrainingCheckPoint(Callback):
  def __init__(self, threshold: Optional[int] = 1, patience: Optional[int] = 5):
    super(TrainingCheckPoint, self).__init__()
    self.threshold = threshold
    self.patience = patience

  def on_epoch_end(self, epoch, logs):
    overfit_patience = 0
    overfit_ratio = logs["val_loss"] / logs["loss"]

    if self.threshold >= overfit_ratio:
      # self.model.save(f"model_{epoch}_{logs['val_accuracy']}.h5", overwrite=False)
      print(f"\ncurrent loss: {logs['loss']}\ncurrent validation loss: {logs['val_loss']}\n Epoch {epoch} was saved with {logs['val_accuracy']} accuracy")
    else:
      overfit_patience += 1
      print(f"Current overfitting epoch count {overfit_patience}")
      if overfit_patience >= self.patience:
        self.model.stop_training = True

```

## Exponential Learning Rate with Epoch Warm-up
The learning rate is scheduled using exponential decay with an initial warm-up phase, allowing a gradual increase in the learning rate for the first few epochs and a slow decay afterward to stabilize training.

```Python
class ExpLRScheduler(Callback):
  def __init__(self, k: Optional[float] = 0.1):
    super(ExpLRScheduler, self).__init__()
    self.k = k

  def schedule_lr(self, epoch, lr):
    # Learning rate warm-up
    if epoch <= 8:
        return lr * math.exp((self.k * 0.125) * epoch)
    # LR exponential decay over k
    else:
        return lr * math.exp(-self.k (epoch / 512))

  def on_epoch_end(self, epoch, logs=None):
    updated_lr = self.schedule_lr(epoch, self.model.optimizer.lr.numpy())
    self.model.optimizer.lr.assign(updated_lr)
    print(f"*** Updated Learning Rate: {updated_lr} for epoch: {epoch + 1}")
```

# Evaluation
[Full Results](https://wandb.ai/alone-wolf/FV%20101%20%7C%20Fine-Tuning%20~%20EfficientNetV2B0_model_0/reports/EfficientNetV2-Transfer-Learning-With-Food-Vision-Dataset-10---Vmlldzo5NTUxOTc5?accessToken=vfv8b3e6x0480airn15748j0bwxzarnd3afemgm1jg10rbnkz03iuwehl22ja1yt)
## Validation Accuracy
![W B Chart 9_30_2024, 3_28_25 PM](https://github.com/user-attachments/assets/5e561655-aa37-4514-8298-5a4c568a9697)

## Validation Loss
![W B Chart 9_30_2024, 3_29_14 PM](https://github.com/user-attachments/assets/2a2d6436-e26a-4818-8e98-62168f64eff5)

## Training Loss
![W B Chart 9_30_2024, 3_30_00 PM](https://github.com/user-attachments/assets/1c7375fb-4ff4-4ff9-8de1-a6af2a29bdc9)

## Learning Rate
![W B Chart 9_30_2024, 3_31_05 PM](https://github.com/user-attachments/assets/4a2ca735-669c-4385-b7a6-53f94f815cb3)

## Hyperparameter Importance
![W B Chart 9_30_2024, 3_29_23 PM](https://github.com/user-attachments/assets/f158f102-dab2-4ac8-9194-c2370bb85e31)

