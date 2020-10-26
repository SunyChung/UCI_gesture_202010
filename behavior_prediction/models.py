import keras
from keras import layers
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model


def build_2018():  # why is it still 50 %??
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=(8, 18, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model


def build_LSTM():
    model = Sequential()
    model.add(LSTM(16, input_shape=(8, 18)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def build_concate():
    # Isn't the Conv2D stride=(1,3) can handle the below hassle?!?!
    left_hand_input = keras.Input(shape=(None,), name='left_hand')
    right_hand_input = keras.Input(shape=(None,), name='right_hand')
    head_input = keras.Input(shape=(None,), name='head')
    spine_input = keras.Input(shape=(None,), name='spine')
    left_wrist_input = keras.Input(shape=(None,), name='left_wrist')
    right_wrist_input = keras.Input(shape=(None,), name='right_wrist')
    person_input = keras.Input(shape=(None,), name='who')

    left_hand_features = Conv2D(16, 3, activation='relu')(left_hand_input)
    right_hand_features = Conv2D(16, 3, activation='relu')(right_hand_input)
    head_features = Conv2D(16, 3, activation='relu')(head_input)
    spine_features = Conv2D(16, 3, activation='relu')(spine_input)
    left_wrist_features = Conv2D(16, 3, activation='relu')(left_wrist_input)
    right_wrist_features = Conv2D(16, 3, activation='relu')(right_wrist_input)
    person_features = LSTM(16)

    x = layers.concatenate([left_hand_features, right_hand_features,
                            head_features, spine_features,
                            left_wrist_features, right_wrist_features,
                            person_features])

    gesture_pred = layers.Dense(5, name='label')(x)

    model = Model(inputs = [left_hand_input, right_hand_input, head_input, spine_input,
                                  left_wrist_input, right_wrist_input, person_input],
                        output = gesture_pred,
                        )
    plot_model(model, 'multi_concate.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
        )
    return model