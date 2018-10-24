import tensorflow as tf;
import numpy as np;
from threading import Thread;
from collections import deque;
import time, os, librosa;
import _pickle as pickle;
from Audio import melspectrogram, _normalize;
from random import shuffle, random;
from Hyper_Parameters import speaker_Embedding_Parameters, synthesizer_Parameters, sound_Parameters;

class Speaker_Embedding:
    def __init__(
        self,
        is_Training = True,
        ):
        self.Placeholder_Generate();
        if is_Training:
            self.Path_Data_Generate();

            self.pattern_Queue = deque();

            pattern_Generate_Thread = Thread(target=self.Pattern_Generate);
            pattern_Generate_Thread.daemon = True;
            pattern_Generate_Thread.start();

            self.is_Test_Pattern_List_Generated = False;
            test_Pattern_List_Generate_Thread = Thread(target=self.test_Pattern_List_Generate);
            test_Pattern_List_Generate_Thread.daemon = True;
            test_Pattern_List_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeHolders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_Training_Placeholder");    #boolean            
            self.placeholder_Dict["Mel"] = tf.placeholder(tf.float32, shape=(None, None, speaker_Embedding_Parameters.mel_Dimension), name="mel_Placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Mel_Length"] = tf.placeholder(tf.int32, shape=(None,), name="mel_Length_Placeholder");    #[batch_Size];
            self.placeholder_Dict["Speaker"] = tf.placeholder(tf.int32, shape=(None,), name="speaker_Placeholder");    #[batch_Size];

    def Path_Data_Generate(self):        
        metadata_Path = os.path.join(speaker_Embedding_Parameters.pattern_Path, "Pattern_Metadata_Dict.pickle").replace("\\", "/");
        with open(metadata_Path, "rb") as f:
            metadata_Dict = pickle.load(f);

        speaker_List = sorted(list(set([speaker for speaker in metadata_Dict["Speaker_Dict"].values()])))        
        speaker_Pattern_Path_List_Dict = {speaker: [] for speaker in speaker_List};
        for pattern_Path, speaker in metadata_Dict["Speaker_Dict"].items():
            speaker_Pattern_Path_List_Dict[speaker].append(pattern_Path)

        self.pattern_Path_List_Dict = {
            speaker: pattern_Path_List
            for speaker, pattern_Path_List in speaker_Pattern_Path_List_Dict.items()
            if len(pattern_Path_List) >= speaker_Embedding_Parameters.batch_Pattern_per_Speaker
            }
        self.speaker_Index_Dict = {speaker: index for index, speaker in enumerate(sorted(list(speaker_Pattern_Path_List_Dict.keys())))}        

    def Pattern_Generate(self):
        speaker_List = list(self.speaker_Index_Dict.keys())

        while True:            
            shuffle(speaker_List);    #Randomized order

            speaker_Batch_List = [speaker_List[x:x+speaker_Embedding_Parameters.batch_Speaker] for x in range(0, len(speaker_List), speaker_Embedding_Parameters.batch_Speaker)]
            #speaker_Batch_List[-1] += speaker_List[:speaker_Embedding_Parameters.batch_Speaker - len(speaker_Batch_List[-1])]
            current_Index = 0;
            while current_Index < len(speaker_Batch_List):
                if len(self.pattern_Queue) >= speaker_Embedding_Parameters.max_Queue:
                    time.sleep(0.1);
                    continue;
                self.New_Pattern_Append(speaker_Batch_List[current_Index]);
                current_Index += 1;

    def New_Pattern_Append(self, speaker_Batch_List):
        pattern_Count = speaker_Embedding_Parameters.batch_Speaker * speaker_Embedding_Parameters.batch_Pattern_per_Speaker;
        mel_Length = np.random.randint(speaker_Embedding_Parameters.pattern_Frame_Range[0], speaker_Embedding_Parameters.pattern_Frame_Range[0] + 1);
        
        mel_Spectrogram_Pattern_Array = np.zeros((pattern_Count, mel_Length, speaker_Embedding_Parameters.mel_Dimension)).astype(np.float32);
        mel_Spectrogram_Length_Array = np.zeros((pattern_Count)).astype(np.int32) + mel_Length;
        speaker_Pattern_Array = np.zeros((pattern_Count)).astype(np.int32);
        
        for speaker_Index, speaker in enumerate(speaker_Batch_List):
            for pattern_Index, pattern_Path in enumerate(sorted(self.pattern_Path_List_Dict[speaker], key=lambda x: random())[:speaker_Embedding_Parameters.batch_Pattern_per_Speaker]):
                pattern_Index = speaker_Index*speaker_Embedding_Parameters.batch_Pattern_per_Speaker + pattern_Index;
                
                with open(os.path.join(speaker_Embedding_Parameters.pattern_Path, pattern_Path).replace("\\", "/"), "rb") as f:
                    load_Dict = pickle.load(f);
                mel = load_Dict["Mel"];
                if speaker != load_Dict["Speaker"]:
                    assert False;
                if mel.shape[0] > mel_Length:
                    cut_Start_Point = np.random.randint(0, mel.shape[0]-mel_Length);
                    cut_End_Point = cut_Start_Point + mel_Length;
                    mel_Spectrogram_Pattern_Array[pattern_Index] = mel[cut_Start_Point:cut_End_Point];
                elif mel.shape[0] == mel_Length:
                    mel_Spectrogram_Pattern_Array[pattern_Index] = mel;
                else:
                    array_Start_Point = np.random.randint(0, mel_Length - mel.shape[0]);
                    array_End_Point = array_Start_Point + mel.shape[0];
                    mel_Spectrogram_Pattern_Array[pattern_Index, array_Start_Point:array_End_Point] = mel;            

                speaker_Pattern_Array[pattern_Index] = self.speaker_Index_Dict[speaker];

        feed_Dict = {
            self.placeholder_Dict["Is_Training"]: True,
            self.placeholder_Dict["Mel"]: _normalize(mel_Spectrogram_Pattern_Array),
            self.placeholder_Dict["Mel_Length"]: mel_Spectrogram_Length_Array,
            self.placeholder_Dict["Speaker"]: speaker_Pattern_Array
            }

        self.pattern_Queue.append(feed_Dict);        
        
    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Queue.popleft();

    def test_Pattern_List_Generate(    
        self,
        wav_Path= "C:/Simulation_Raw_Data.Temp/VoxCeleb1/Test/wav",
        ):
        using_Extension = [x.upper() for x in [".wav", ".m4a", ".flac"]];
        batch_Size = speaker_Embedding_Parameters.batch_Speaker * speaker_Embedding_Parameters.batch_Pattern_per_Speaker;
        mel_Length = int(np.mean(speaker_Embedding_Parameters.pattern_Frame_Range));
        minimum_Length = mel_Length * (4 / 2 + .5)

        
        file_Path_Dict = {}
        for root, directory_List, file_Name_List in os.walk(wav_Path):
            speaker = root.replace("\\", "/").split("/")[-2];
            if not speaker in file_Path_Dict.keys():
                file_Path_Dict[speaker] = [];
            file_Path_Dict[speaker].extend([os.path.join(root, file_Name).replace("\\", "/") for file_Name in file_Name_List]);

        label_List = [];
        pattern_List = [];

        used_Speaker = 0;
        for speaker, wav_File_Path_List in file_Path_Dict.items():
            if used_Speaker >= 10:
                break;
            if len(wav_File_Path_List) < 50:
                continue;
            used_File = 0;
            for wav_File_Path in wav_File_Path_List:
                if used_File >= 50:
                    break;
                if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                    continue;

                signal = librosa.core.load(wav_File_Path, sr = sound_Parameters.sample_Rate)[0].astype(np.float32);
                signal *= .99 / np.max(np.abs(signal));     #Normalize                        
                new_Pattern = np.transpose(melspectrogram(
                    y= signal,
                    num_freq= sound_Parameters.spectrogram_Dimension,
                    frame_shift_ms= sound_Parameters.mel_Frame_Shift,
                    frame_length_ms= sound_Parameters.mel_Frame_Length,
                    num_mels= speaker_Embedding_Parameters.mel_Dimension,
                    sample_rate= sound_Parameters.sample_Rate
                    )).astype(np.float32);

                if new_Pattern.shape[0] < minimum_Length:
                    continue;

                for pattern_Start_Index in range(0, new_Pattern.shape[0], mel_Length):
                    slice_Pattern = new_Pattern[pattern_Start_Index:pattern_Start_Index+mel_Length];
                    if slice_Pattern.shape[0] < mel_Length:
                        break;
                    label_List.append((speaker, used_File));
                    pattern_List.append(slice_Pattern);
                
                used_File += 1;
            if used_File > 0:
                used_Speaker += 1;
                            
        self.label_List = label_List;
        self.test_Pattern_List = [];
        for x in range(0, len(pattern_List), batch_Size):
            batch_Pattern_List = pattern_List[x:x+batch_Size]
            new_Pattern_Feed = {   
                self.placeholder_Dict["Is_Training"]: False,
                self.placeholder_Dict["Mel"]: _normalize(np.stack(batch_Pattern_List, 0)),
                self.placeholder_Dict["Mel_Length"]: np.zeros((len(batch_Pattern_List))).astype(np.int32) + mel_Length
                }
            self.test_Pattern_List.append(new_Pattern_Feed);
        self.is_Test_Pattern_List_Generated = True;

    def Get_Test_Pattern_List(self):
        while not self.is_Test_Pattern_List_Generated:
            time.sleep(1);
        return self.label_List, self.test_Pattern_List;

    def Get_Mel_Feed_from_Voice(self, wav_Path):    #This will be used with TTS model, not speaker embedding training.
        mel_Length = int(np.mean(speaker_Embedding_Parameters.pattern_Frame_Range));

        signal = librosa.core.load(wav_Path, sr = sound_Parameters.sample_Rate)[0].astype(np.float32);
        signal *= .99 / np.max(np.abs(signal));     #Normalize
                        
        new_Pattern = np.transpose(melspectrogram(
            y= signal,
            num_freq= sound_Parameters.spectrogram_Dimension,
            frame_shift_ms= sound_Parameters.mel_Frame_Shift,
            frame_length_ms= sound_Parameters.mel_Frame_Length,
            num_mels= speaker_Embedding_Parameters.mel_Dimension,
            sample_rate= sound_Parameters.sample_Rate
            )).astype(np.float32);
        
        pattern_List = [];

        if new_Pattern.shape[0] < mel_Length:
            array_Start_Point = np.random.randint(0, mel_Length - new_Pattern.shape[0]);
            array_End_Point = array_Start_Point + new_Pattern.shape[0];
            padding_Pattern = np.zeros((mel_Length, speaker_Embedding_Parameters.mel_Dimension)).astype(np.float32);
            padding_Pattern[array_Start_Point:array_End_Point] = new_Pattern;
            sample_Pattern = padding_Pattern;
        
        for pattern_Start_Index in range(0, new_Pattern.shape[0], mel_Length):
            slice_Pattern = new_Pattern[pattern_Start_Index:pattern_Start_Index+mel_Length];
            if slice_Pattern.shape[0] < mel_Length:
                break;
            pattern_List.append(slice_Pattern);
                    
        new_Pattern_Feed = {   
            self.placeholder_Dict["Is_Training"]: False,
            self.placeholder_Dict["Mel"]: _normalize(np.stack(pattern_List, 0)),
            self.placeholder_Dict["Mel_Length"]: np.zeros((len(pattern_List))).astype(np.int32) + mel_Length
            }

        return new_Pattern_Feed


class Synthesizer:
    def __init__(
        self,
        is_Training = True,
        ):
        self.Placeholder_Generate();

        if is_Training:
            self.Path_Data_Generate();

            self.pattern_Queue = deque();

            pattern_Generate_Thread = Thread(target=self.Pattern_Generate);
            pattern_Generate_Thread.daemon = True;
            pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeHolders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_Training_Placeholder");    #boolean            
            self.placeholder_Dict["Token"] = tf.placeholder(tf.int32, shape=(None, None), name="token_Placeholder");    #Shape: [batch_Size, token_Length];
            self.placeholder_Dict["Token_Length"] = tf.placeholder(tf.int32, shape=(None), name="token_Length_Placeholder");    #Shape: [batch_Size];
            self.placeholder_Dict["Speaker_Embedding"] = tf.placeholder(tf.float32, shape=(None, speaker_Embedding_Parameters.embedding_Size), name="speaker_Embedding_Placeholder");    #[batch_Size, speaker_Embedding_Size];
            self.placeholder_Dict["Mel"] = tf.placeholder(tf.float32, shape=(None, None, sound_Parameters.tts_Mel_Dimension), name="mel_Spectrogram_Placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Mel_Length"] = tf.placeholder(tf.int32, shape=(None,), name="mel_Spectrogram_Length_Placeholder");    #[batch_Size];
       
    def Path_Data_Generate(self):        
        metadata_Path = os.path.join(synthesizer_Parameters.pattern_Path, "Pattern_Metadata_Dict.pickle").replace("\\", "/");
        with open(metadata_Path, "rb") as f:
            metadata_Dict = pickle.load(f);
        self.pattern_Path_List = sorted(list([path for path, mel_Length in metadata_Dict["TTS_Mel_Length_Dict"].items() if mel_Length < synthesizer_Parameters.decoder_Max_Mel_Length]));
        self.token_Length_List = [metadata_Dict["Text_Length_Dict"][path] for path in self.pattern_Path_List];

        self.token_Index_Dict = {token: index for index, token in enumerate(metadata_Dict["Token_List"])};
        self.index_Token_Dict = {index: token for index, token in enumerate(metadata_Dict["Token_List"])};
        self.token_Count = len(metadata_Dict["Token_List"]);

    def Pattern_Generate(self):
        path_Index_List = list(range(len(self.pattern_Path_List)));        
        if synthesizer_Parameters.pattern_Sorting:
            path_Index_List = [x for _, x in sorted(zip(self.token_Length_List, path_Index_List))]    #Sequence by length
            
        while True:            
            path_Index_Batch_List = [path_Index_List[x:x+synthesizer_Parameters.batch_Size] for x in range(0, len(path_Index_List), synthesizer_Parameters.batch_Size)]
            shuffle(path_Index_Batch_List);

            current_Index = 0;
            while current_Index < len(path_Index_Batch_List):
                if len(self.pattern_Queue) >= synthesizer_Parameters.max_Queue:
                    time.sleep(0.1);
                    continue;
                self.New_Pattern_Append(path_Index_Batch_List[current_Index]);
                current_Index += 1;

    def New_Pattern_Append(self, path_Index_List):
        pattern_Count = len(path_Index_List);

        token_Pattern_List = [];
        speaker_Embedding_List = [];
        mel_Pattern_List = [];
        for path_Index in path_Index_List:
            with open(os.path.join(synthesizer_Parameters.pattern_Path, self.pattern_Path_List[path_Index]).replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f);
            token_Pattern_List.append(self.Text_to_Token_Index(load_Dict["Text"]));
            speaker_Embedding_List.append(load_Dict["Speaker_Embedding"][np.random.randint(10)]);
            mel_Pattern_List.append(load_Dict["TTS_Mel"]);
                    
        max_Token_Pattern_Length = max([x.shape[0] for x in token_Pattern_List]);
        max_Mel_Pattern_Length = int(np.ceil(max([x.shape[0] for x in mel_Pattern_List]) / synthesizer_Parameters.decoder_Output_Size_per_Step) * synthesizer_Parameters.decoder_Output_Size_per_Step);
        
        token_Pattern_Array = np.zeros((pattern_Count, max_Token_Pattern_Length)).astype(np.int32);
        token_Length_Array = np.stack([x.shape[0] for x in token_Pattern_List], axis= 0).astype(np.int32);
        speaker_Embedding_Array = np.vstack(speaker_Embedding_List).astype(np.float32);
        mel_Pattern_Array = np.zeros((pattern_Count, max_Mel_Pattern_Length, sound_Parameters.tts_Mel_Dimension)).astype(np.float32);
        mel_Length_Array = np.stack([x.shape[0] for x in mel_Pattern_List], axis= 0).astype(np.int32);
        
        for pattern_Index, (token_Pattern, mel_Pattern) in enumerate(zip(token_Pattern_List, mel_Pattern_List)):
            token_Pattern_Array[pattern_Index, :token_Pattern.shape[0]] = token_Pattern;
            mel_Pattern_Array[pattern_Index, :mel_Pattern.shape[0]] = mel_Pattern;
                    
        feed_Dict = {
            self.placeholder_Dict["Is_Training"]: True,            
            self.placeholder_Dict["Token"]: token_Pattern_Array,
            self.placeholder_Dict["Token_Length"]: token_Length_Array,
            self.placeholder_Dict["Speaker_Embedding"]: speaker_Embedding_Array,
            self.placeholder_Dict["Mel"]: mel_Pattern_Array,
            self.placeholder_Dict["Mel_Length"]: mel_Length_Array,
            }

        self.pattern_Queue.append(feed_Dict);

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Queue.popleft();
    
    def Get_Test_Pattern(self, test_Set_List):  #Test_Set: Tuple (test_String, speaker embedding pattern)
        pattern_Count = len(test_Set_List);

        token_Pattern_List = [];
        speaker_Embedding_List = [];
        for test_String, speaker_Embedding_Pattern in test_Set_List:
            token_Pattern_List.append(self.Text_to_Token_Index(test_String));
            speaker_Embedding_List.append(speaker_Embedding_Pattern);
                    
        max_Token_Pattern_Length = max([x.shape[0] for x in token_Pattern_List]);
        
        token_Pattern_Array = np.zeros((pattern_Count, max_Token_Pattern_Length)).astype(np.int32);
        token_Length_Array = np.stack([x.shape[0] for x in token_Pattern_List], axis= 0).astype(np.int32);
        speaker_Embedding_Array = np.vstack(speaker_Embedding_List).astype(np.float32);

        max_Iterations = int(np.ceil(synthesizer_Parameters.decoder_Max_Mel_Length / synthesizer_Parameters.decoder_Output_Size_per_Step));        
        dummy_Mel_Pattern_Array = np.zeros((pattern_Count, max_Iterations, sound_Parameters.tts_Mel_Dimension)).astype("float32");
        dummy_Mel_Length_Array = np.zeros((pattern_Count)).astype("int32");
        
        for pattern_Index, token_Pattern in enumerate(token_Pattern_List):
            token_Pattern_Array[pattern_Index, :token_Pattern.shape[0]] = token_Pattern;

        feed_Dict = {
            self.placeholder_Dict["Is_Training"]: False,
            self.placeholder_Dict["Token"]: token_Pattern_Array,
            self.placeholder_Dict["Token_Length"]: token_Length_Array,
            self.placeholder_Dict["Speaker_Embedding"]: speaker_Embedding_Array,
            self.placeholder_Dict["Mel"]: dummy_Mel_Pattern_Array,
            self.placeholder_Dict["Mel_Length"]: dummy_Mel_Length_Array,
            }

        return feed_Dict

    def Text_to_Token_Index(self, text):
        text = text.lower()
        if set(self.token_Index_Dict.keys()) != set(text) | set(self.token_Index_Dict.keys()):
            print("Inserted text: {}\nThere is a letter which is not compatible. This letter will be removed.".format(text))
        return np.stack([self.token_Index_Dict[token] for token in text.lower() if token in self.token_Index_Dict.keys()]);

    def Token_Index_to_Text(self, token_Pattern):
        return "".join([self.index_Token_Dict[index] for index in token_Pattern]);

if __name__ == "__main__":
    new_Synthesizer = Synthesizer();

    with open(os.path.join(synthesizer_Parameters.pattern_Path, "LS.0012354.pickle").replace("\\", "/"), "rb") as f:
        load_Dict1 = pickle.load(f);
    with open(os.path.join(synthesizer_Parameters.pattern_Path, "VCTK.0005002.pickle").replace("\\", "/"), "rb") as f:
        load_Dict2 = pickle.load(f);

    test_Set_List = [
        ("He wants to buy a car.", load_Dict1["Speaker_Embedding"][np.random.randint(10)]),
        ("He wants to buy a car.", load_Dict2["Speaker_Embedding"][np.random.randint(10)]),
        ]
    #print(new_Synthesizer.Get_Test_Pattern(test_Set_List=test_Set_List));

    print(new_Synthesizer.Get_Pattern());