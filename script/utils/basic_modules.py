from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D
from keras.layers import Dropout, Flatten, Input
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.layers import core
from keras.layers import Layer


def ed(x):
    return tf.expand_dims(x, axis=1)


def cc(x):
    return tf.concat([x[0], x[1], x[2], x[3], x[4], x[5]], axis=1)


def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()  # ||x||^2
    scale = K.sqrt(s_quared_norm) / (0.5 + s_quared_norm)  # ||x||/(0.5+||x||^2)
    result = scale * x
    return result


# 定义自己的softmax函数
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result


class Capsule(Layer):
    def __init__(self,
                 num_capsule=100,
                 dim_capsule=100,
                 routings=1,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)  # Capsule继承**kwargs参数
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)  # 得到激活函数

    # 定义权重
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            # 自定义权重
            self.kernel = self.add_weight(  # [row,col,channel]->[1,input_dim_capsule,num_capsule*dim_capsule]
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)  # 必须继承Layer的build方法

    # 层的功能逻辑(核心)
    def call(self, inputs):
        if self.share_weights:
            # inputs: [batch, input_num_capsule, input_dim_capsule]
            # kernel: [1, input_dim_capsule, num_capsule*dim_capsule]
            # hat_inputs: [batch, input_num_capsule, num_capsule*dim_capsule]
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        # hat_inputs: [batch, input_num_capsule, num_capsule, dim_capsule]
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        # hat_inputs: [batch, num_capsule, input_num_capsule, dim_capsule]
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        # b: [batch, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                b += K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):  # 自动推断shape
        return (None, self.num_capsule, self.dim_capsule)


def RPI_EDLCN(encoders_pro, encoders_rna, p_s_f_l1, r_s_f_l1, p_s_f_l2, r_s_f_l2, p_s_f_l3, r_s_f_l3, vector_repeatition_cnn):

    if type(vector_repeatition_cnn) == int:
        vec_len_p = vector_repeatition_cnn
        vec_len_r = vector_repeatition_cnn
    else:
        vec_len_p = vector_repeatition_cnn[0]
        vec_len_r = vector_repeatition_cnn[1]

    # CNN
    xp_in_cnn = Input(shape=(p_s_f_l1, vec_len_p))
    xp_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xp_in_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xp_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xp_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Flatten()(xp_cnn)
    xp_out1 = Dense(16)(xp_cnn)
    xp_out1 = Dropout(0.2)(xp_out1)
    xp_out1 = core.Lambda(ed)(xp_out1)

    xr_in_cnn = Input(shape=(r_s_f_l1, vec_len_r))
    xr_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xr_in_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Flatten()(xr_cnn)
    xr_out1 = Dense(16)(xr_cnn)
    xr_out1 = Dropout(0.2)(xr_out1)
    xr_out1 = core.Lambda(ed)(xr_out1)

    #  DNN
    xp_in_dnn = Input(shape=(p_s_f_l2, vec_len_p))  # 500,574
    xp_dnn = Flatten()(xp_in_dnn)
    xp_dnn = Dense(units=500, input_dim=574, use_bias=True, activation='relu', name='dense1')(xp_dnn)
    xp_dnn = Dropout(rate=0.5)(xp_dnn)
    xp_dnn = Dense(units=256, use_bias=True, activation='tanh')(xp_dnn)
    xp_dnn = Dropout(rate=0.5)(xp_dnn)
    xp_dnn = Dense(units=128, use_bias=True, activation='tanh')(xp_dnn)
    xp_dnn = Dropout(rate=0.5)(xp_dnn)
    xp_out2 = Dense(16)(xp_dnn)
    xp_out2 = Dropout(0.2)(xp_out2)
    xp_out2 = core.Lambda(ed)(xp_out2)

    xr_in_dnn = Input(shape=(r_s_f_l2, vec_len_r))  # 2000, 3178
    xr_dnn = Flatten()(xr_in_dnn)
    xr_dnn = Dense(units=2000, input_dim=3178, use_bias=True, activation='relu', name='dense2')(xr_dnn)
    xr_dnn = Dropout(rate=0.5)(xr_dnn)
    xr_dnn = Dense(units=256, use_bias=True, activation='tanh')(xr_dnn)
    xr_dnn = Dropout(rate=0.5)(xr_dnn)
    xr_dnn = Dense(units=128, use_bias=True, activation='tanh')(xr_dnn)
    xr_dnn = Dropout(rate=0.5)(xr_dnn)
    xr_out2 = Dense(16)(xr_dnn)
    xr_out2 = Dropout(0.2)(xr_out2)
    xr_out2 = core.Lambda(ed)(xr_out2)

    # SAE
    xp_in_sae = Input(shape=(p_s_f_l3,))
    xp_encoded = encoders_pro[0](xp_in_sae)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoded = encoders_pro[1](xp_encoded)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoder = encoders_pro[2](xp_encoded)
    xp_encoder = Dropout(0.2)(xp_encoder)
    xp_encoder = BatchNormalization()(xp_encoder)
    xp_encoder = Dropout(0.2)(xp_encoder)
    xp_out3 = Dense(16)(xp_encoder)
    xp_out3 = Dropout(0.2)(xp_out3)
    xp_out3 = core.Lambda(ed)(xp_out3)

    xr_in_sae = Input(shape=(r_s_f_l3,))
    xr_encoded = encoders_rna[0](xr_in_sae)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[1](xr_encoded)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[2](xr_encoded)
    xr_encoder = Dropout(0.2)(xr_encoded)
    xr_encoder = BatchNormalization()(xr_encoder)
    xr_encoder = Dropout(0.2)(xr_encoder)
    xr_out3 = Dense(16)(xr_encoder)
    xr_out3 = Dropout(0.2)(xr_out3)
    xr_out3 = core.Lambda(ed)(xr_out3)

    x_out = core.Lambda(cc)([xp_out1, xr_out1, xp_out2, xr_out2, xp_out3, xr_out3])

    #y_cnn_dnn_sae = Flatten()(x_out)
    #y_cnn_dnn_sae = Dropout(0.5)(y_cnn_dnn_sae)
    #y_cnn_dnn_sae = Dense(2, activation='softmax')(y_cnn_dnn_sae)
    #model_RPI_EDLCN = Model(inputs=[xp_in_cnn, xr_in_cnn, xp_in_dnn, xr_in_dnn, xp_in_conjoint, xr_in_conjoint], outputs=y_cnn_dnn_sae)
    y_cnn_dnn_sae = Capsule(num_capsule=100, dim_capsule=100,  routings=1, share_weights=True)(x_out)
    y_cnn_dnn_sae = Flatten()(y_cnn_dnn_sae)
    y_cnn_dnn_sae = Dropout(0.5)(y_cnn_dnn_sae)
    y_cnn_dnn_sae = Dense(2, activation='softmax')(y_cnn_dnn_sae)
    model_RPI_EDLCN = Model(inputs=[xp_in_cnn, xr_in_cnn, xp_in_dnn, xr_in_dnn, xp_in_sae, xr_in_sae], outputs=y_cnn_dnn_sae)

    return model_RPI_EDLCN
