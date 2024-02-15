import time
import tensorflow as tf
from tensorflow.keras import layers
from network.att import AttentionBlock

class FourDirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = layers.Bidirectional(layers.LSTM(self.hidden_size, return_sequences=True))
        self.lstm_downup = layers.Bidirectional(layers.LSTM(self.hidden_size, return_sequences=True))

    def build(self, input_shape):
        # Implement build method if needed for custom layer initialization
        pass

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x_rightleft = tf.reshape(x, (batch_size, self.seq_size, self.feat_size))
        x_downup = tf.transpose(x_rightleft, perm=[0, 2, 1])
        hidden_rightleft = self.lstm_rightleft(x_rightleft)
        hidden_downup = self.lstm_downup(x_downup)
        hlr_fw = hidden_rightleft[:, :, :self.hidden_size]
        hlr_bw = hidden_rightleft[:, :, self.hidden_size:]
        hud_fw = hidden_downup[:, :, :self.hidden_size]
        hud_bw = hidden_downup[:, :, self.hidden_size:]
        return tf.concat([hlr_fw[:, -1, :], hlr_bw[:, -1, :], hud_fw[:, -1, :], hud_bw[:, -1, :]], axis=1)

class AtLoc(tf.keras.Model):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, lstm=False):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        self.inference_time = None

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.layers[-1] = layers.GlobalAveragePooling2D()
        fe_out_planes = self.feature_extractor.layers[-1].output_shape[-1]
        self.feature_extractor.layers[-1] = layers.Dense(feat_dim)

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xy = layers.Dense(2)
            self.fc_yaw = layers.Dense(1)
        else:
            self.att = AttentionBlock(feat_dim)
            self.fc_xy = layers.Dense(2)
            self.fc_yaw = layers.Dense(1)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.layers[-1], self.fc_xy, self.fc_yaw]
        else:
            init_modules = self.submodules()

        for m in init_modules:
            if isinstance(m, layers.Conv2D) or isinstance(m, layers.Dense):
                m.build((None, fe_out_planes))
                m.kernel_initializer = 'glorot_normal'
                if m.bias is not None:
                    m.bias_initializer = 'zeros'

    def call(self, x):
        tstart = time.time()

        x = self.feature_extractor(x)
        x = tf.nn.relu(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(tf.shape(x)[0], -1))

        if self.droprate > 0:
            x = tf.nn.dropout(x, rate=self.droprate)

        xy = self.fc_xy(x)
        yaw = self.fc_yaw(x)

        self.inference_time = time.time() - tstart

        return tf.concat([xy, yaw], axis=1)

    def get_last_inference_time(self, with_nms=True):
        res = [self.inference_time]
        return res
