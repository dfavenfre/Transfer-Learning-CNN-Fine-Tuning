def view_random_images(
    target_dir,
    target_class
    ):
  target_folder = os.path.join(target_dir, target_class)
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])

  plt.imshow(img)
  # plt.title(target_class + " with a shape of: " + str(img.shape))
  plt.axis("off")

  return img

def resize_images(
    file_path: str,
    img_shape: int = 224
    ):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_image(image)
  image = tf.image.resize(image,
                          size=[img_shape, img_shape]
                          )
  image = tf.expand_dims(image / 255., axis=0)

  return image


def unzip_data(file_path: str):
  zip_ref = zipfile.ZipFile(file_path)
  zip_ref.extractall()
  zip_ref.close()


def plot_predictions(
    target_dir: str,
    model
    ):
  class_labels = os.listdir("10_food_classes_all_data/train/")
  target_class = random.choice(class_labels)
  target_folder = os.path.join(target_dir, target_class)

  random_image = random.choice(os.listdir(target_folder))
  image_path = os.path.join(target_folder,
                            random_image)
  resized_image = resize_images(image_path)

  prediction = model.predict(resized_image)
  prediction_object = class_labels[int(tf.round(prediction))]

  image = mpimg.imread(image_path)
  plt.imshow(image)
  plt.title(f"Predicted Object: {prediction_object}")
  plt.axis(False)
  plt.show()


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


def optimize_network():

  global train_data
  global test_data
  global sweep_config

  tf.keras.backend.clear_session()
  with wandb.init(config=sweep_config):

    config = wandb.config

    if config.optimizers=="adam":
      optimizer = tf.keras.optimizers.Adam(
          learning_rate = config.lr
          )
    elif config.optimizers=="sgd":
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=config.lr
      )
    else:
      raise NotImplementedError(
          f"Provided Optimizer :{config.optimizers} Not Implement"
          )
    model = train_model_0()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            name="Categorical_Loss"
            ),
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    training_history = model.fit(
        train_data,
        validation_data=(test_data),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=[
            WandbMetricsLogger(),
            ExpLRScheduler(k=config.k_value),
            TrainingCheckPoint(
                patience=config.early_stopping_patience)
        ]
    )


  return training_history
