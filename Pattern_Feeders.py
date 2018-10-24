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
