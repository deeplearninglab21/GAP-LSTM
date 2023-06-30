"""
Original version used in the paper GAP-LSTM.
"""

import tensorflow as tf
import numpy as np
import logging
import sys

from lib.spektral_utilities import *
from lib.spektral_gcn import GraphConv

logger_name = __name__
logger = logging.getLogger(logger_name)
logger.setLevel("INFO")


class GAP_LSTM(tf.keras.Model):
    def __init__(self, h, p, adj, nodes, n_features, **kwargs):
        """
        Positional Arguments:
        h: history steps
        p: prediction steps
        adj: adjacency matrix
        nodes: number of nodes
        n_features: number of features

        Keyword Arguments:
        cell_type: String - type of the recurrent cell. Possible values are gclstm, nomemorystate, lstm (default: gclstm)
        conv: Bool - whether the 2D convolution is active (default: True)
        attention: Bool - whether the attention mechanism is active (default: True)
        """
        super(GAP_LSTM, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.logger.warning(
            __name__
            + " initalizing with: \n\th: "
            + str(h)
            + "\n\tp: "
            + str(p)
            + "\n\tf:"
            + str(n_features)
            + "\n\tadj shape: "
            + str(adj.shape)
        )

        self.cell_type = kwargs.get("cell_type", "gclstm")  # Unused
        self.has_conv = kwargs.get("conv", True)
        self.has_attention = kwargs.get("attention", True)
        self.gcn = kwargs.get("gcn", "spektral")

        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(h, nodes, n_features), dtype=tf.float64
        )

        self.start_ff = FeedForward([n_features])
        self.sagl = SA_GCLSTM(
            h,
            p,
            nodes,
            n_features,
            self.cell_type,
            self.has_conv,
            self.has_attention,
            self.gcn,
        )
        self.end_ff = FeedForward([1])

        self.adj = adj
        self.adj_weights = tf.Variable(
            initial_value=tf.ones(tf.TensorShape(self.adj.shape)),
            trainable=True,
            shape=tf.TensorShape(self.adj.shape),
        )

    def call(self, inputs):
        adj = tf.math.multiply(self.adj, self.adj_weights)
        x = inputs  # b,t,n,f
        x = self.input_layer(x)

        self.logger.info(
            __name__
            + " called with shapes:\n\tx: "
            + str(inputs.shape)
            + "\n\tadj: "
            + str(adj.shape)
        )

        x = self.start_ff(x)
        x = self.sagl([x, adj])
        x = self.end_ff(x)  # x: (b,t,n,1)
        x = tf.squeeze(x, axis=-1)  # x: (b,t,n)

        return x

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}


class SA_GCLSTM(tf.keras.layers.Layer):
    def __init__(
        self,
        h,
        p,
        n,
        f,
        cell_type="gclstm",
        has_conv=True,
        has_attention=True,
        gcn="spektral",
    ):
        super(SA_GCLSTM, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.logger.info("{} initalizing with: \n\tf: {}".format(__name__, f))

        self.has_conv = has_conv

        self.encoder = Encoder(n, f, gcn, cell_type=cell_type)
        self.decoder = Decoder(
            h, p, n, f, gcn, cell_type=cell_type, has_attention=has_attention
        )

        if self.has_conv:
            self.conv_filters = p
            self.conv = tf.keras.layers.Conv2D(
                self.conv_filters, (2, 2), data_format="channels_first", padding="same"
            )

        self.log_encoder_states = tf.Variable(
            initial_value=tf.fill([0, h, n, f], value=1 / h),
            trainable=False,
            validate_shape=False,
            shape=[None, h, n, f],
        )
        self.log_decoder_states = tf.Variable(
            initial_value=tf.fill([0, p, n, f], value=1 / h),
            trainable=False,
            validate_shape=False,
            shape=[None, p, n, f],
        )

    def call(self, inputs, training=None):
        x, adj = inputs

        enc_hidden_states, enc_last_h, enc_last_c, enc_last_m = self.encoder([x, adj])
        dec_hidden_states = self.decoder(
            [enc_last_h, enc_last_c, enc_last_m, enc_hidden_states, adj]
        )

        if not training:
            self.log_encoder_states.assign(
                tf.concat([self.log_encoder_states, enc_hidden_states], axis=0)
            )
            self.log_decoder_states.assign(
                tf.concat([self.log_decoder_states, dec_hidden_states], axis=0)
            )

        output = dec_hidden_states
        if self.has_conv:
            conv_output = self.conv(tf.expand_dims(enc_last_h, axis=1))
            output = dec_hidden_states + conv_output

        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, nodes, n_features, gcn, cell_type="gclstm"):
        super(Encoder, self).__init__()
        self.logger = logging.getLogger(logger_name)

        self.cell = None
        if cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(n_features, gcn)
        elif cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(n_features, gcn, has_memory_state=False)
        else:
            self.cell = None  # Raise error

    def call(self, inputs):
        x, adj = inputs
        b, h, n, f = x.shape

        last_h = tf.ones((b, n, f))
        last_c = tf.ones((b, n, f))
        last_m = tf.ones((b, n, f))

        hidden_states = []
        for i in range(h):
            if isinstance(self.cell, tf.keras.layers.LSTMCell):
                last_h, [_, last_c] = self.cell(
                    tf.reshape(x[:, i, :, :], [b, n * f]),
                    [tf.reshape(last_h, [b, n * f]), tf.reshape(last_c, [b, n * f])],
                )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical
                self.logger.critical(
                    "LSTMCell output: next_hidden_state: {}, next_cell_state: {}".format(
                        last_h.shape, last_c.shape
                    )
                )
                last_m = tf.zeros_like(last_h)
            else:
                last_h, last_c, last_m = self.cell(
                    [x[:, i, :, :], last_h, last_c, last_m, adj]
                )

            hidden_states.append(last_h)  # [h,[B,N,F]]

        hidden_states = tf.stack(hidden_states, axis=1)
        return hidden_states, last_h, last_c, last_m


class MultiheadAttention(tf.keras.layers.Layer):
    """
    Multihead attention with a head for each node.
    If we have N nodes, it returns N maps each containing a representation of F for the respective nodo.
    Query matrix Q contains only the current decoder step, and key and value matrices K, V contain all encoder steps.
    For a batch size B, history steps H, nodes N and features F, inputs must have dimensions:
    Q : [B,N,F], K : [B,H,N,F], V : [B,H,N,F], where
    H: encoder steps
    P: decoder steps
    N: number of nodes
    F: number of features
    """

    def __init__(self, H, P, N, F):
        super(MultiheadAttention, self).__init__()
        self.logger = logging.getLogger(logger_name)

        self.logger.info(
            "MultiheadAttention initializing with:\n\tH: "
            + str(H)
            + "\n\tP: "
            + str(P)
            + "\n\tN: "
            + str(N)
            + "\n\tF: "
            + str(F)
        )
        self.H = H
        self.P = P
        self.N = N
        self.F = F

        self.Wq = [
            FeedForward([F]) for n in range(N)
        ]  # takes a single step, so weights are [F,F]
        self.Wk = [
            [FeedForward([F]) for h in range(H)] for n in range(N)
        ]  # takes H steps, so weights are [H,F,F]
        self.Wv = [
            [FeedForward([F]) for h in range(H)] for n in range(N)
        ]  # takes H steps, so weights are [H,F,F]

    def call(self, inputs):
        # Q :   [B,N,F]
        # K : [B,H,N,F]
        # V : [B,H,N,F]
        Q, K, V = inputs
        self.logger.info(
            "MultiheadAttention called with:\n\tQ shape: "
            + str(Q.shape)
            + "\n\tK shape: "
            + str(K.shape)
            + "\n\tV shape: "
            + str(V.shape)
        )

        heads = []
        log_scores = []
        for n in range(self.N):  # Head n
            Qn = Q[:, n, :]  # take node n,   [B,F]
            Kn = K[:, :, n, :]  # take node n, [B,H,F]
            Vn = V[:, :, n, :]  # take node n, [B,H,F]

            # Attention(Qn, Kn, Vn) = softmax(Qn*Wqn * (Kn*Wkn)^T) * (Vn*Wvn)
            QWq = self.Wq[n](Qn)  # [B,F] x [F,F] -> [B,F]

            KWk = [
                self.Wk[n][h](Kn[:, h]) for h in range(self.H)
            ]  # [H,[B,F]] x [H,[F,F]] -> [H,[B,F]]
            VWv = [
                self.Wv[n][h](Vn[:, h]) for h in range(self.H)
            ]  # [H,[B,F]] x [H,[F,F]] -> [H,[B,F]]
            KWk = tf.stack(KWk)  # [H,B,F]
            VWv = tf.stack(VWv)  # [H,B,F]

            KWk = tf.transpose(KWk, perm=[1, 2, 0])  # [B,F,H]
            VWv = tf.transpose(VWv, perm=[1, 0, 2])  # [B,H,F]

            scores = tf.nn.softmax(
                tf.linalg.matmul(tf.expand_dims(QWq, axis=1), KWk)
            )  # [B,1,F] x [B,F,H] -> [B,1,H]
            log_scores.append(scores)  # [n,[B,1,H]]

            result = tf.linalg.matmul(scores, VWv)  # [B,1,H] x [B,H,F] -> [B,1,F]
            result = tf.squeeze(result, axis=[1])  # [B,F]
            heads.append(result)  # [n,[B,F]]

        log_scores = tf.stack(log_scores, axis=1)  # [B,N,1,H]
        log_scores = tf.squeeze(log_scores, axis=2)  # [B,N,H]

        heads = tf.stack(heads)  # [N,B,F]
        heads = tf.transpose(heads, perm=[1, 0, 2])  # [B,N,F]
        return heads, log_scores


class Decoder(tf.keras.layers.Layer):
    def __init__(self, h, p, n, f, gcn, cell_type="gclstm", has_attention=True):
        super(Decoder, self).__init__()
        self.logger = logging.getLogger(logger_name)

        self.h = h
        self.p = p
        self.n = n
        self.f = f

        self.cell = None
        if cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(f, gcn)
        elif cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(f, gcn, has_memory_state=False)
        elif cell_type == "lstm":
            self.cell = tf.keras.layers.LSTMCell(n * f)
        else:
            self.cell = None  # Raise error

        self.attention = None
        if has_attention:
            self.attention = MultiheadAttention(h, p, n, f)
            self.attw = tf.Variable(
                initial_value=tf.fill([0, p, n, h], value=1 / h),
                trainable=False,
                validate_shape=False,
                shape=[None, p, n, h],
            )

    def call(self, inputs, training=None):
        # hidden_states: [B,H,N,F]
        (
            last_hidden_state,
            last_cell_state,
            last_memory_state,
            hidden_states,
            adj,
        ) = inputs

        dec_result = []
        batch_scores = []
        for i in range(self.p):
            if self.attention:
                if isinstance(self.cell, tf.keras.layers.LSTMCell):
                    last_hidden_state = tf.reshape(last_hidden_state, [b, 1, -1])
                    hidden_states = tf.reshape(hidden_states, [b, h, 1, -1])
                decoder_input, scores = self.attention(
                    [last_hidden_state, hidden_states, hidden_states]
                )
                batch_scores.append(scores)  # [p,[B,N,H]]
            else:
                decoder_input = last_hidden_state

            if isinstance(self.cell, tf.keras.layers.LSTMCell):
                next_hidden_state, [_, next_cell_state] = self.cell(
                    tf.reshape(decoder_input, [b, -1]),
                    [
                        tf.reshape(last_hidden_state, [b, -1]),
                        tf.reshape(last_cell_state, [b, -1]),
                    ],
                )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical
                self.logger.critical(
                    "Decoder LSTMCell output: next_hidden_state: {}, next_cell_state: {}".format(
                        next_hidden_state.shape, next_cell_state.shape
                    )
                )
                next_memory_state = tf.zeros_like(next_hidden_state)
            else:
                next_hidden_state, next_cell_state, next_memory_state = self.cell(
                    [
                        decoder_input,
                        last_hidden_state,
                        last_cell_state,
                        last_memory_state,
                        adj,
                    ]
                )  # all [B,N,F]

            last_hidden_state = next_hidden_state
            last_cell_state = next_cell_state
            last_memory_state = next_memory_state

            dec_result.append(next_hidden_state)  # [p,[B,N,F]]

        if self.attention and not training:
            batch_scores = tf.stack(batch_scores, axis=1)  # [B,P,N,H]
            self.attw.assign(tf.concat([self.attw, batch_scores], axis=0))

        dec_output = tf.stack(dec_result, axis=1)  # [B,P,N,F]
        return dec_output


class GCNLSTM_Cell(tf.keras.layers.Layer):
    def __init__(self, n_features, gcn, has_memory_state=True):
        super(GCNLSTM_Cell, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.logger.info(
            "GCNLSTM_Cell initalizing with: \n\tn_features: " + str(n_features)
        )

        self.has_memory_state = has_memory_state

        if gcn == "spektral":
            self.fx_gcn = GraphConv(n_features, activation="relu")
            self.fh_gcn = GraphConv(n_features, activation="relu")
            self.ix_gcn = GraphConv(n_features, activation="relu")
            self.ih_gcn = GraphConv(n_features, activation="relu")
            self.cx_gcn = GraphConv(n_features, activation="relu")
            self.ch_gcn = GraphConv(n_features, activation="relu")
            self.ox_gcn = GraphConv(n_features, activation="relu")
            self.oh_gcn = GraphConv(n_features, activation="relu")
            if has_memory_state:
                self.i_gcn = GraphConv(n_features, activation="relu")
                self.g_gcn = GraphConv(n_features, activation="relu")
                self.o_gcn = GraphConv(n_features, activation="relu")
                self.im_gcn = GraphConv(n_features, activation="relu")
                self.gm_gcn = GraphConv(n_features, activation="relu")
                self.om_gcn = GraphConv(n_features, activation="relu")
        else:
            self.fx_gcn = WeightedGraphConv(n_features, activation="relu")
            self.fh_gcn = WeightedGraphConv(n_features, activation="relu")
            self.ix_gcn = WeightedGraphConv(n_features, activation="relu")
            self.ih_gcn = WeightedGraphConv(n_features, activation="relu")
            self.cx_gcn = WeightedGraphConv(n_features, activation="relu")
            self.ch_gcn = WeightedGraphConv(n_features, activation="relu")
            self.ox_gcn = WeightedGraphConv(n_features, activation="relu")
            self.oh_gcn = WeightedGraphConv(n_features, activation="relu")
            if has_memory_state:
                self.i_gcn = WeightedGraphConv(n_features, activation="relu")
                self.g_gcn = WeightedGraphConv(n_features, activation="relu")
                self.o_gcn = WeightedGraphConv(n_features, activation="relu")
                self.im_gcn = WeightedGraphConv(n_features, activation="relu")
                self.gm_gcn = WeightedGraphConv(n_features, activation="relu")
                self.om_gcn = WeightedGraphConv(n_features, activation="relu")

    def call(self, inputs):
        x, hidden_state, cell_state, memory_state, adj = inputs

        self.logger = logging.getLogger(logger_name)
        self.logger.debug(
            "Calling GCNLSTM_Cell with shapes: \n\tx: {}\n\th: {}\n\tc: {}\n\tm: {}\n\tadj: {}".format(
                x.shape,
                hidden_state.shape,
                cell_state.shape,
                memory_state.shape,
                adj.shape,
            )
        )

        fx = self.fx_gcn([x, adj])
        fh = self.fh_gcn([hidden_state, adj])
        ix = self.ix_gcn([x, adj])
        ih = self.ih_gcn([hidden_state, adj])
        cx = self.cx_gcn([x, adj])
        ch = self.ch_gcn([hidden_state, adj])
        ox = self.ox_gcn([x, adj])
        oh = self.oh_gcn([hidden_state, adj])

        f = tf.math.sigmoid(fx + fh)
        i = tf.math.sigmoid(ix + ih)
        o = tf.math.sigmoid(ox + oh)
        c = f * cell_state + i * tf.math.tanh(cx + ch)
        h = o * tf.math.tanh(c)

        if self.has_memory_state:
            SA_ih = self.i_gcn([h, adj])
            SA_im = self.im_gcn([memory_state, adj])
            SA_gh = self.g_gcn([h, adj])
            SA_gm = self.gm_gcn([memory_state, adj])
            SA_oh = self.o_gcn([h, adj])
            SA_om = self.om_gcn([memory_state, adj])

            i = tf.math.sigmoid(SA_ih + SA_im)
            g = tf.math.sigmoid(SA_gh + SA_gm)
            o = tf.math.sigmoid(SA_oh + SA_om)

            self.logger.debug(
                "Gates computed with shapes: "
                + "\n\ti: "
                + str(i.shape)
                + "\n\tg: "
                + str(g.shape)
                + "\n\to: "
                + str(o.shape)
            )

            m = i * memory_state + (1 - i) * g
            h = m * o

        self.logger.debug(
            "Outputs of the GCLSTM block computed with shape: \n\th: {}\n\tc: {}\n\tm: {}".format(
                c.shape,
                h.shape,
                m.shape if self.has_memory_state else "no memory state",
            )
        )

        return h, c, m if self.has_memory_state else tf.zeros_like(h)  # tutti [B,N,F]


class WeightedGraphConv(tf.keras.layers.Layer):
    def __init__(self, n_features):
        super(GraphConvNetwork, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.logger.info(
            "GraphConvNetwork initalizing with: \n\tn_features: " + str(n_features)
        )

        out_channels = 2 * n_features
        self.dense = FeedForward([out_channels])

    def call(self, inputs):
        x, adj = inputs
        x = tf.matmul(adj, x)
        out = self.dense(x)
        ls, rs = tf.split(out, 2, axis=-1)
        out = ls * tf.math.sigmoid(rs)
        return out


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, sizes, normalize=False):
        """
        sizes: array containing the number of neurons for each FF layer.
        """
        super(FeedForward, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.logger.debug(
            "FeedForward initalizing with: \n\tsizes: "
            + str(sizes)
            + "\n\tnormalize: "
            + str(normalize)
        )

        self.normalize = normalize
        self.dense_layers = [tf.keras.layers.Dense(units=size) for size in sizes]
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, center=False, scale=False
        )

    def call(self, inputs):
        x = inputs

        self.logger.debug("FF layer called with input shape: " + str(x.shape))

        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i != len(self.dense_layers) - 1:
                x = tf.nn.relu(x)
        if self.normalize:
            x += inputs
            x = self.layer_norm(x)

        self.logger.debug("FF layer output shape: " + str(x.shape))

        return x
