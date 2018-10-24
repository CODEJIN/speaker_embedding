#Basic input pattern shape: (Batch, Letter_Index);
#Embedded input pattern shape: (Batch, Letter_Index, distributed_Pattern)

import tensorflow as tf;
import numpy as np;
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell, LSTMCell, GRUCell, MultiRNNCell, LSTMStateTuple, OutputProjectionWrapper;
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauMonotonicAttention, BasicDecoder, dynamic_decode, Helper, TrainingHelper;
from ZoneoutLSTMCell import ZoneoutLSTMCell;      
from Hyper_Parameters import synthesizer_Parameters, sound_Parameters;

def Cosine_Similarity(x,y):
    """
    Compute the cosine similarity between same row of two tensors.
    Args:
        x: nd tensor (...xMxN).
        y: nd tensor (...xMxN). A tensor of the same shape as x
    Returns:        
        cosine_Similarity: A (n-1)D tensor representing the cosine similarity between the rows. Size is (...xM)
    """
    return tf.reduce_sum(x * y, axis=-1) / (tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1)));

def Cosine_Similarity2D(x, y):
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:        
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1]);   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1]);   #[M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 2)) + 1e-8)  #[M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity");

    return cosine_Similarity;

def Batch_Cosine_Similarity2D(x, y):    
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to y's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:        
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (BATCH x M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1]);   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1]);   #[Batch, M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 3)) + 1e-8)  #[Batch, M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity");

    return cosine_Similarity;


def Conv1D(input_Pattern, scope, is_Training, kernel_Size, filter_Count, activation):
    with tf.variable_scope(scope):
        conv1D_Output = tf.layers.conv1d(
            input_Pattern,
            filters=filter_Count,
            kernel_size=kernel_Size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1D_Output, training = is_Training);

def Prenet(input_Pattern, layer_Size_List=[256] * 2, dropout_Rate=0.5, is_Training=True, scope="prenet"):
    prenet_Activation = input_Pattern;
    with tf.variable_scope(scope):
        for index, layer_Size in enumerate(layer_Size_List):
            prenet_Activation = tf.layers.dropout(
                tf.layers.dense(
                    prenet_Activation,
                    layer_Size,
                    activation = tf.nn.relu,
                    use_bias=True,
                    name = "activation_{}".format(index)
                ),
                rate = dropout_Rate,
                training = is_Training,
                name = "dropout_{}".format(index)
            )

    return prenet_Activation;


def Encoder(
    input_Pattern,  #[Batch, Embedding_Dimension]
    input_Length,   #[Batch]
    speaker_Embedding_Pattern,  #[Batch, Speaker_Embedding]
    is_Training = True,
    scope = "encoder_Module"
    ):
    with tf.variable_scope(scope):
        conv_Activation = input_Pattern;
        for index, (filter_Count, kernel_Size) in enumerate([(synthesizer_Parameters.encoder_Conv_Filter_Count, synthesizer_Parameters.encoder_Conv_Kernel_Size)] * synthesizer_Parameters.encoder_Conv_Layer_Count):
            conv_Activation = Conv1D(
                input_Pattern = conv_Activation,
                filter_Count = filter_Count,
                kernel_Size = kernel_Size,
                activation = tf.nn.relu,
                scope = "conv_%d" % index,
                is_Training = is_Training
                )

        #Bidirectional LSTM
        output_Pattern_List, rnn_State_List = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = ZoneoutLSTMCell(
                synthesizer_Parameters.encoder_LSTM_Cell_Size, 
                is_training=is_Training, 
                cell_zoneout_rate=synthesizer_Parameters.encoder_Zoneout_Rate, 
                output_zoneout_rate=synthesizer_Parameters.encoder_Zoneout_Rate
                ),
            cell_bw = ZoneoutLSTMCell(
                synthesizer_Parameters.encoder_LSTM_Cell_Size, 
                is_training=is_Training, 
                cell_zoneout_rate=synthesizer_Parameters.encoder_Zoneout_Rate, 
                output_zoneout_rate=synthesizer_Parameters.encoder_Zoneout_Rate
                ),
            inputs = conv_Activation,
            sequence_length = input_Length,
            dtype = tf.float32,
            scope = "biLSTM"
        )

        speaker_Embedded_BiLSTM_Activation = tf.concat(list(output_Pattern_List) + [tf.tile(tf.expand_dims(speaker_Embedding_Pattern, axis=1), multiples=(1, tf.shape(output_Pattern_List[0])[1], 1))], axis=2);

        return speaker_Embedded_BiLSTM_Activation;
    

def Decoder(
    batch_Size,
    attention_Mechanism,    
    is_Training = True,
    target_Pattern = None,
    scope = "decoder_Module"
    ):
    with tf.variable_scope(scope):
        decoder_Prenet_Cell = DecoderPrenetWrapper(
            layer_Size_List= [synthesizer_Parameters.decoder_Prenet_Layer_Size] * synthesizer_Parameters.decoder_Prenet_Layer_Count,
            dropout_Rate= synthesizer_Parameters.decoder_Prenet_Dropout_Rate,
            is_Training = is_Training
            )

        attention_Cell = AttentionWrapper(
            cell = decoder_Prenet_Cell,
            attention_mechanism = attention_Mechanism,
            alignment_history = True,
            output_attention = False,
            name = "attention"
            )

        concat_Cell = ConcatOutputAndAttentionWrapper(cell = attention_Cell);   #256 + 128 = 384

        decoder_Cell = MultiRNNCell(
            cells = [
                concat_Cell,
                ZoneoutLSTMCell(
                    synthesizer_Parameters.decoder_LSTM_Cell_Size, 
                    is_training=is_Training, 
                    cell_zoneout_rate= synthesizer_Parameters.decoder_Zoneout_Rate, 
                    output_zoneout_rate= synthesizer_Parameters.decoder_Zoneout_Rate
                    ),
                ZoneoutLSTMCell(
                    synthesizer_Parameters.decoder_LSTM_Cell_Size, 
                    is_training= is_Training, 
                    cell_zoneout_rate= synthesizer_Parameters.decoder_Zoneout_Rate, 
                    output_zoneout_rate= synthesizer_Parameters.decoder_Zoneout_Rate
                    )
                ]
            )
        projection_Cell = LinearProjectionWrapper(
            cell = decoder_Cell,
            linear_Projection_Size = sound_Parameters.tts_Mel_Dimension * synthesizer_Parameters.decoder_Output_Size_per_Step, 
            stop_Token_Size = 1
            )

        decoder_Initial_State = projection_Cell.zero_state(batch_size=batch_Size, dtype=tf.float32);    
    
        helper = Tacotron2_Helper(
            is_Training = is_Training,
            batch_Size = batch_Size,
            target_Pattern = target_Pattern,
            output_Dimension = sound_Parameters.tts_Mel_Dimension,
            output_Size_per_Step = synthesizer_Parameters.decoder_Output_Size_per_Step,
            linear_Projection_Size = sound_Parameters.tts_Mel_Dimension * synthesizer_Parameters.decoder_Output_Size_per_Step,
            stop_Token_Size = 1
        )

        final_Outputs, final_States, final_Sequence_Lengths = dynamic_decode(
            decoder = BasicDecoder(projection_Cell, helper, decoder_Initial_State),
            maximum_iterations = int(np.ceil(synthesizer_Parameters.decoder_Max_Mel_Length / synthesizer_Parameters.decoder_Output_Size_per_Step)),
        )
        linear_Projection_Activation, stop_Token = tf.split(
            final_Outputs.rnn_output, 
            num_or_size_splits=[sound_Parameters.tts_Mel_Dimension * synthesizer_Parameters.decoder_Output_Size_per_Step, 1], 
            axis=2
            )

        linear_Projection_Activation = tf.reshape(linear_Projection_Activation, [batch_Size, -1, sound_Parameters.tts_Mel_Dimension]);    
        alignment_Histroy = final_States[0].alignment_history;

        return linear_Projection_Activation, tf.squeeze(stop_Token, axis=2), alignment_Histroy.stack();


def Postnet(
    input_Pattern,    
    is_Training = True,
    scope = "postnet"
    ):    
    with tf.name_scope(name=scope):
        conv_Activation = input_Pattern;
        for index, (filter_Count, kernel_Size) in enumerate([(synthesizer_Parameters.decoder_Postnet_Conv_Filter_Count, synthesizer_Parameters.decoder_Postnet_Conv_Kernal_Size)] * synthesizer_Parameters.decoder_Postnet_Conv_Layer_Count):
            conv_Activation = Conv1D(
                input_Pattern = conv_Activation,
                filter_Count = filter_Count,
                kernel_Size = kernel_Size,
                #activation = tf.nn.tanh if index < synthesizer_Parameters.decoder_Postnet_Conv_Layer_Count - 1 else None,
                activation = tf.nn.tanh,
                scope = "conv_%d" % index,
                is_Training = is_Training
                )
        correction_Activation = tf.layers.dense(
            conv_Activation,
            units = input_Pattern.get_shape()[2],
            activation = None,
            use_bias = True,
            name = "correction"
            )
    return correction_Activation;


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(self, layer_Size_List=[256]*2, dropout_Rate=0.5, is_Training=True):
        super(DecoderPrenetWrapper, self).__init__();
        self._layer_Size_List=layer_Size_List;
        self._dropout_Rate=dropout_Rate;
        self._is_Training = is_Training;        

    @property
    def state_size(self):
        return self._layer_Size_List[-1]

    @property
    def output_size(self):
        return self._layer_Size_List[-1]

    def call(self, inputs, state):
        prenet_Out = Prenet(
            input_Pattern = inputs,
            layer_Size_List = self._layer_Size_List,
            dropout_Rate = self._dropout_Rate,
            scope = "prenet",
            is_Training = self._is_Training
        )

        return prenet_Out, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros(shape=(batch_size, self._layer_Size_List[-1]))


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class LinearProjectionWrapper(RNNCell):
    '''Projecting the mel-spectrogram and stop token.'''
    def __init__(self, cell, linear_Projection_Size, stop_Token_Size):
        super(LinearProjectionWrapper, self).__init__()
        self._cell = cell
        self._linear_Projection_Size = linear_Projection_Size;
        self._stop_Token_Size = stop_Token_Size;

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._linear_Projection_Size + self._stop_Token_Size;

    def call(self, inputs, state):
        outputs, res_state = self._cell(inputs, state);
        projection = tf.layers.dense(
            outputs,
            units = self._linear_Projection_Size + self._stop_Token_Size,
            activation = None,
            use_bias = True,
            name = "mel_Projection"
        )
        mel_Projection, stop_Token = tf.split(projection, num_or_size_splits=[self._linear_Projection_Size, self._stop_Token_Size], axis=1);
        stop_Token = tf.nn.sigmoid(stop_Token);

        return tf.concat([mel_Projection, stop_Token], axis=1), res_state;

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class Tacotron2_Helper(Helper):
    def __init__(self, is_Training, batch_Size, target_Pattern, output_Dimension, output_Size_per_Step, linear_Projection_Size, stop_Token_Size):
        '''
            batch_Size: batch size
            target_Pattern: decoder Mel-Spectrogram Target
            output_Dimension: Mel-Spectrogram dimension (ex. 80)
            output_Size_Per_Step: For last index
        '''
        with tf.name_scope('Tacotron2_Helper'):
            # inputs is [N, T_in], targets is [N, T_out, D]
            self.is_Training = is_Training;
            self.batch_Size = batch_Size;
            self.output_Dimension = output_Dimension;
            self.linear_Projection_Size = linear_Projection_Size;
            self.stop_Token_Size = stop_Token_Size;

            #Only Training use.
            # Feed every r-th target frame as input
            self.target_Pattern_as_Input = target_Pattern[:, output_Size_per_Step-1::output_Size_per_Step, :];
            # Use full length for every target because we don't want to mask the padding frames
            num_Steps = tf.shape(self.target_Pattern_as_Input)[1]
            self.length = tf.tile([num_Steps], [self.batch_Size])    # All batch have same length.

    @property
    def batch_size(self):
        return self.batch_Size;

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([]);  #Ignored

    @property
    def sample_ids_dtype(self):
        return np.int32;

    def initialize(self, name=None):
        return (tf.tile([False], [self.batch_Size]), _go_frames(self.batch_Size, self.output_Dimension));

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self.batch_Size]);  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope('Tacotron2_Helper'):
            def is_Training_True():
                finished = (time + 1 >= self.length)
                
                #Rayhane's code assign the predicted pattern with probability. See the commented code. I ignore that.
                next_Input = self.target_Pattern_as_Input[:, time, :]   #Teacher_forcing                
                #next_inputs = tf.cond(
				#tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
				#lambda: self._targets[:, time, :], #Teacher-forcing: return true frame
				#lambda: outputs[:,-self._output_dim:])

                return (finished, next_Input, state)

            def is_Training_False():
                linear_Projection, stop_Token = tf.split(outputs, num_or_size_splits=[self.linear_Projection_Size, self.stop_Token_Size], axis=1);
                
                # Feed last output frame as next input. outputs is [N, output_dim * r]
                next_Input = linear_Projection[:, -self.output_Dimension:]                
                # When stop_Token is over 0.5, model stop.
                finished = tf.cast(tf.round(tf.squeeze(stop_Token, axis=1)), dtype=tf.bool);

                return (finished, next_Input, state)

            return tf.cond(pred = self.is_Training, true_fn = is_Training_True, false_fn = is_Training_False);

def _go_frames(batch_Size, output_Dimension):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_Size, output_Dimension])