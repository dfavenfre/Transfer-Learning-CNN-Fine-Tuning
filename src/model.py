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