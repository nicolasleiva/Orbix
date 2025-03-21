import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Capa que añade codificación posicional a la entrada.
    """
    def __init__(self, d_model, dropout_rate=0.1, max_seq_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        return self.dropout(x, training=training)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Capa del Transformer para la codificación.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
    Capa del Transformer para la decodificación.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False):
        attn1 = self.mha1(x, x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, enc_output, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

class TransformerTrajectoryModel(tf.keras.Model):
    """
    Modelo Transformer para predecir trayectorias orbitales.
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, dff=2048, dropout_rate=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(3)  # Salidas: x, y, z

    def call(self, inputs, training=False):
        x = self.pos_encoder(inputs, training=training)
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        for decoder in self.decoder_layers:
            x = decoder(x, x, training=training)
        return self.final_layer(x)
