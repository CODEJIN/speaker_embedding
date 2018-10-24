import numpy as np;
import tensorflow as tf;
from tensorflow.contrib.seq2seq import BasicDecoder, TrainingHelper, dynamic_decode;
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, ResidualWrapper;
from Customized_Modules import Cosine_Similarity, Cosine_Similarity2D, Batch_Cosine_Similarity2D;
from Pattern_Feeders import Speaker_Embedding as SE_Feeder;
from Hyper_Parameters import speaker_Embedding_Parameters, sound_Parameters;
import time, os, argparse;
from sklearn.manifold import TSNE;
import matplotlib.pyplot as plt;

class SE_Model:
    def __init__(self, is_Training = True):
        self.is_Training = is_Training;

        self.pattern_Feeder = SE_Feeder(is_Training= is_Training);
        self.tf_Session = tf.Session();
        
        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5);        

    def Tensor_Generate(self):
        placeholder_Dict = self.pattern_Feeder.placeholder_Dict;
        
        with tf.variable_scope('speaker_Embedding') as scope:
            batch_Size = tf.shape(placeholder_Dict["Mel"])[0];

            input_Activation =  tf.layers.dense(
                inputs= placeholder_Dict["Mel"],
                units= speaker_Embedding_Parameters.embedding_Size
                )

            rnn_Cell = MultiRNNCell(cells = [
                ResidualWrapper(MultiRNNCell(cells = [
                    ResidualWrapper(LSTMCell(num_units= 768, num_proj = speaker_Embedding_Parameters.embedding_Size, activation=tf.nn.tanh)),
                    ResidualWrapper(LSTMCell(num_units= 768, num_proj = speaker_Embedding_Parameters.embedding_Size, activation=tf.nn.tanh)),
                ])),
                LSTMCell(num_units= 768, num_proj = speaker_Embedding_Parameters.embedding_Size, activation=tf.nn.tanh),
                ])

            helper = TrainingHelper(
                inputs= input_Activation,
                sequence_length = placeholder_Dict["Mel_Length"],
                time_major = False
                )

            decoder_Initial_State = rnn_Cell.zero_state(batch_size=batch_Size, dtype=tf.float32);    

            final_Outputs, final_States, final_Sequence_Lengths = dynamic_decode(
                decoder = BasicDecoder(rnn_Cell, helper, decoder_Initial_State),
                maximum_iterations = speaker_Embedding_Parameters.pattern_Frame_Range[1],
                )

            #hidden_Activation = tf.nn.sigmoid(final_Outputs.rnn_output[:, -1, :]);
            hidden_Activation = final_Outputs.rnn_output[:, -1, :];
            embedding_Activation = tf.nn.l2_normalize(hidden_Activation, axis=1);

            self.averaged_Embedding_Tensor = tf.reduce_mean(embedding_Activation, axis=0);  #For single wav

            if not self.is_Training:
                self.tf_Session.run(tf.global_variables_initializer());                
                return;

        #Back-prob.
        with tf.variable_scope('training_Loss') as scope:            
            speaker_Size = tf.cast(batch_Size / speaker_Embedding_Parameters.batch_Pattern_per_Speaker, tf.int32);
                        
            reshaped_Embedding_Activation = tf.reshape(
                embedding_Activation,
                shape=(
                    speaker_Size,
                    speaker_Embedding_Parameters.batch_Pattern_per_Speaker,
                    speaker_Embedding_Parameters.embedding_Size,
                    )     #[speaker, pattern_per_Speaker, embedding]
                )
                                
            centroid_for_Within = (tf.tile(
                tf.reduce_sum(reshaped_Embedding_Activation, axis=1, keepdims=True),    #[speaker, 1, embedding]
                multiples= [1,speaker_Embedding_Parameters.batch_Pattern_per_Speaker,1] #[speaker, pattern_per_Speaker, embedding]
                ) - reshaped_Embedding_Activation) / (speaker_Embedding_Parameters.batch_Pattern_per_Speaker - 1)   #[speaker, pattern_per_Speaker, embedding]
            centroid_for_Between = tf.reduce_mean(reshaped_Embedding_Activation, axis=1)    #[speaker, embedding]            
                
            cosine_Similarity_Weight = tf.Variable(10.0, name='cosine_Similarity_Weight', trainable = True);
            cosine_Similarity_Bias = tf.Variable(-5.0, name='cosine_Similarity_Bias', trainable = True);

            within_Cosine_Similarity = cosine_Similarity_Weight * Cosine_Similarity(reshaped_Embedding_Activation, centroid_for_Within) - cosine_Similarity_Bias  #[speaker, pattern_per_Speaker]
            
            between_Cosine_Similarity_Filter = 1 - tf.tile(
                tf.expand_dims(tf.eye(speaker_Size), axis=1),
                multiples=[1, speaker_Embedding_Parameters.batch_Pattern_per_Speaker, 1]
                )  #[speaker, pattern_per_Speaker, Speaker]
            between_Cosine_Similarity = tf.reshape(
                cosine_Similarity_Weight * Cosine_Similarity2D(embedding_Activation, centroid_for_Between) - cosine_Similarity_Bias,    #[speaker * pattern_per_Speaker, speaker]
                shape=(
                    speaker_Size,
                    speaker_Embedding_Parameters.batch_Pattern_per_Speaker,
                    speaker_Size,
                    )
                )     #[speaker, pattern_per_Speaker, Speaker]

            between_Cosine_Similarity = tf.reshape(
                tf.boolean_mask(between_Cosine_Similarity, between_Cosine_Similarity_Filter),
                shape = (
                    speaker_Size,
                    speaker_Embedding_Parameters.batch_Pattern_per_Speaker,
                    speaker_Size - 1,
                    )
                )   #[speaker, pattern_per_Speaker, Speaker - 1]     Same speaker of first dimension was removed at last dimension.

            ##softmax_Loss = within_Cosine_Similarity - tf.log(tf.reduce_sum(tf.exp(tf.concat([tf.expand_dims(within_Cosine_Similarity, axis=2), between_Cosine_Similarity], axis=2)), axis = 2));
            softmax_Loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits= tf.concat([tf.expand_dims(within_Cosine_Similarity, axis=2), between_Cosine_Similarity], axis=2),
                labels= tf.zeros(shape=(speaker_Size, speaker_Embedding_Parameters.batch_Pattern_per_Speaker), dtype=tf.int32)
                )   #Almost same

            contrast_Loss = 1 - tf.nn.sigmoid(within_Cosine_Similarity) + tf.reduce_max(between_Cosine_Similarity, axis = 2);

            if speaker_Embedding_Parameters.loss_Method.upper() == "Softmax".upper():
                loss = tf.reduce_mean(softmax_Loss);
            elif speaker_Embedding_Parameters.loss_Method.upper() == "Contrast".upper():
                loss = tf.reduce_mean(contrast_Loss);

            global_Step = tf.Variable(0, name='global_Step', trainable = False);

            #Noam decay of learning rate
            step = tf.cast(global_Step + 1, dtype=tf.float32);
            warmup_Steps = 4000.0;
            learning_Rate = speaker_Embedding_Parameters.learning_Rate * warmup_Steps ** 0.5 * tf.minimum(step * warmup_Steps**-1.5, step**-0.5);

            #Weight update. We use the ADAM optimizer
            optimizer = tf.train.AdamOptimizer(learning_Rate);
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)
            
        self.training_Tensor_List = [global_Step, learning_Rate, loss, optimize];
        self.test_Tensor_List = [global_Step, embedding_Activation];
        
        if not os.path.exists(speaker_Embedding_Parameters.extract_Path + "/Summary"):
            os.makedirs(speaker_Embedding_Parameters.extract_Path + "/Summary");
        graph_Writer = tf.summary.FileWriter(speaker_Embedding_Parameters.extract_Path + "/Summary", self.tf_Session.graph);
        graph_Writer.close();
        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):
        if not os.path.exists(speaker_Embedding_Parameters.extract_Path + "/Checkpoint"):
            os.makedirs (speaker_Embedding_Parameters.extract_Path + "/Checkpoint");

        checkpoint_Path = tf.train.latest_checkpoint(speaker_Embedding_Parameters.extract_Path + "/Checkpoint");
        print("Lastest checkpoint:", checkpoint_Path);

        if checkpoint_Path is None:
            print("There is no checkpoint");
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print("Checkpoint '", checkpoint_Path, "' is loaded");

    def Train(
        self,
        test_Step = 1000,
        checkpoint_Step = 1000,
        ):
        if not os.path.exists(speaker_Embedding_Parameters.extract_Path + "/Checkpoint"):
            os.makedirs(speaker_Embedding_Parameters.extract_Path + "/Checkpoint");

        self.Test();
        try:
            while True:
                start_Time = time.time();
                global_Step, learning_Rate, loss, _ = self.tf_Session.run(
                    fetches = self.training_Tensor_List,
                    feed_dict = self.pattern_Feeder.Get_Pattern()
                )
                print(
                    "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                    "Global_Step:", global_Step, "\t",
                    "Learning_Rate:", learning_Rate, "\t",
                    "Loss:", loss
                )

                if (global_Step % checkpoint_Step) == 0:
                    self.tf_Saver.save(self.tf_Session, speaker_Embedding_Parameters.extract_Path + '/Checkpoint/Checkpoint', global_step=global_Step)
                    print("Checkpoint saved");

                if (global_Step % test_Step) == 0:
                    self.Test();

        except KeyboardInterrupt:
            self.tf_Saver.save(self.tf_Session, speaker_Embedding_Parameters.extract_Path + '/Checkpoint/Checkpoint', global_step=global_Step)
            print("Checkpoint saved");
            self.Test();

    def Test(self):
        if not os.path.exists(speaker_Embedding_Parameters.extract_Path):
            os.makedirs(speaker_Embedding_Parameters.extract_Path);        

        test_Result_List = [];
        pattern_Label_List, test_Pattern_Feed_List = self.pattern_Feeder.Get_Test_Pattern_List()
        for test_Pattern_Feed in test_Pattern_Feed_List:
            global_Step, test_Result = self.tf_Session.run(
                fetches = self.test_Tensor_List,
                feed_dict = test_Pattern_Feed
            )
            test_Result_List.append(test_Result);
        
        test_Result_Array = np.vstack(test_Result_List);
        
        mean_Label_List = list(set(pattern_Label_List))
        mean_Embedding_Result_Array = np.vstack([np.mean(test_Result_Array[[x==speaker and y==file_Index for x, y in pattern_Label_List]], axis = 0) for speaker, file_Index in mean_Label_List])
        speaker_Label_List = [speaker for speaker, _, in mean_Label_List];

        self.Extract(global_Step, speaker_Label_List, mean_Embedding_Result_Array);

    def Extract(self, global_Step,  speaker_Label_List, mean_Embedding_Result_Array):               
        #Embedding t-SNE
        #https://www.scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
        display_Speaker_List = list(set(speaker_Label_List))[:10];
        color_List = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'];        
        
        tsne = TSNE(n_components=2, random_state=0);
        embedding_Test_Result_2d = tsne.fit_transform(mean_Embedding_Result_Array); #[Pattern, 2]        
        fig = plt.figure(figsize=(10, 10));
        for speaker, color in zip(display_Speaker_List, color_List):
            plt.scatter(
                x= embedding_Test_Result_2d[[speaker == x for x in speaker_Label_List], 0],
                y= embedding_Test_Result_2d[[speaker == x for x in speaker_Label_List], 1],
                c= color,
                edgecolors = 'k',
                label= speaker)
        plt.legend()
        plt.savefig(
            speaker_Embedding_Parameters.extract_Path + "/TSNE.Step_{:010d}.png".format(global_Step),
            bbox_inches='tight'
            )
        plt.close(fig)

    def Get_Embedding(self, wav_Path):                
        embedding_Result = self.tf_Session.run(
            fetches = self.averaged_Embedding_Tensor,
            feed_dict = self.pattern_Feeder.Get_Mel_Feed_from_Voice(wav_Path= wav_Path)
            )

        return embedding_Result;

if __name__ == "__main__":
    new_SE_Model = SE_Model(is_Training= True);
    new_SE_Model.Restore()
    new_SE_Model.Train(test_Step= 1000, checkpoint_Step= 1000)