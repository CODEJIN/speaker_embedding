import numpy as np;
import re, os, librosa, argparse;
from Audio import *;
import _pickle as pickle;
from concurrent.futures import ProcessPoolExecutor as PE;
from Hyper_Parameters import sound_Parameters, speaker_Embedding_Parameters;
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

using_Extension = [x.upper() for x in [".wav", ".m4a", ".flac"]];

'''
wav파일과 speaker 정보를 pickle로 만듭니다.
각 pickle은 2가지 데이터를 포함합니다: Speaker, 40Mel
Metadata.pickle은 Speaker 정보와 전체 pickle의 세부 정보, pickle list를 포함합니다.
'''
max_Worker = 15;

def Pickle_Generate(lexicon_Name, pattern_Index, speaker, wav_File_Path):    
    new_Pattern_Dict = {};
    new_Pattern_Dict["Speaker"] = speaker;

    signal = librosa.core.load(wav_File_Path, sr = sound_Parameters.sample_Rate)[0].astype(np.float32);
    signal *= .99 / np.max(np.abs(signal));     #Normalize
            
    new_Pattern_Dict["Mel"] = np.transpose(melspectrogram(
        y= signal,
        num_freq= sound_Parameters.spectrogram_Dimension,
        frame_shift_ms= sound_Parameters.mel_Frame_Shift,
        frame_length_ms= sound_Parameters.mel_Frame_Length,
        num_mels= speaker_Embedding_Parameters.mel_Dimension,
        sample_rate= sound_Parameters.sample_Rate
        )).astype(np.float32);

    pattern_File_Name = "{}.{:07d}.pickle".format(lexicon_Name, pattern_Index)
    with open(os.path.join(speaker_Embedding_Parameters.pattern_Path, pattern_File_Name).replace("\\", "/"), "wb") as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2);
            
    print("{}    {}    ->    {}".format(lexicon_Name, wav_File_Path, pattern_File_Name));

def Pattern_Generate_VCTK(
    wav_Path= "D:/Simulation_Raw_Data/VCTK/wav48"
    ):
    if not os.path.exists(speaker_Embedding_Parameters.pattern_Path):
        os.makedirs(speaker_Embedding_Parameters.pattern_Path);
        
    print("VCTK raw file list generating...");
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(wav_Path):
        speaker = "VCTK." + root.replace("\\", "/").split("/")[-1];
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace("\\", "/");
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;            
            file_List.append((speaker, wav_File_Path))
    print("VCTK raw file list generating...Done");

    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, "VCTK", pattern_Index, speaker, wav_File_Path);

def Pattern_Generate_LS(
    data_Path= "D:/Simulation_Raw_Data/LibriSpeech/train"
    ):
    if not os.path.exists(speaker_Embedding_Parameters.pattern_Path):
        os.makedirs(speaker_Embedding_Parameters.pattern_Path);

    print("LS raw file list generating...");
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(data_Path):
        speaker, _ = root.replace("\\", "/").split("/")[-2:];
        speaker = "LS." + speaker;
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace("\\", "/");
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            file_List.append((speaker, wav_File_Path))
    print("LS raw file list generating...Done");

    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, "LS", pattern_Index, speaker, wav_File_Path);

def Pattern_Generate_VC(   
    lexicon_Suffix,    
    wav_Path= "D:/Simulation_Raw_Data/VoxCeleb1/wav"
    ):
    if not os.path.exists(speaker_Embedding_Parameters.pattern_Path):
        os.makedirs(speaker_Embedding_Parameters.pattern_Path);            
        
    print("VC{} raw file list generating...".format(lexicon_Suffix));
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(wav_Path):
        speaker = "VC{}.".format(lexicon_Suffix) + root.replace("\\", "/").split("/")[-2];
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace("\\", "/");
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            file_List.append((speaker, wav_File_Path))
    print("VC{} raw file list generating...Done".format(lexicon_Suffix));
            
    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, "VC{}".format(lexicon_Suffix), pattern_Index, speaker, wav_File_Path);

def Metadata_Generate():    
    speaker_Dict = {};
    mel_Length_Dict = {};
    
    #ProcessPoolExcuter는 별개의 클라이언트로 구동시키기 때문에 global 변수의 사용이 불가능해서 재로드해야됨....
    print("Pickle data check...")
    for root, directory_List, file_Name_List in os.walk(speaker_Embedding_Parameters.pattern_Path):
        for index, pattern_File_Name in enumerate(file_Name_List):
            if pattern_File_Name == "Pattern_Metadata_Dict.pickle":
                continue;
            with open(os.path.join(root, pattern_File_Name).replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f);
            speaker_Dict[pattern_File_Name] = load_Dict["Speaker"];            
            mel_Length_Dict[pattern_File_Name] = load_Dict["Mel"].shape[0];
            print("{}/{}    {}    Done".format(index + 1, len(file_Name_List), pattern_File_Name));

    print("Pickle data check...Done")

    metadata_Dict = {};

    metadata_Dict["Speaker_Dict"] = speaker_Dict;
    metadata_Dict["Mel_Length_Dict"] = mel_Length_Dict;
    
    #Hyper parameters for consistency
    metadata_Dict["Sample_Rate"] = sound_Parameters.sample_Rate;
    metadata_Dict["Frame_Shift"] = sound_Parameters.mel_Frame_Shift;
    metadata_Dict["Frame_Length"] = sound_Parameters.mel_Frame_Length;
    metadata_Dict["Mel_Dimension"] = speaker_Embedding_Parameters.mel_Dimension;
    metadata_Dict["Spectrogram_Dimension"] = sound_Parameters.spectrogram_Dimension;
    
    with open(os.path.join(speaker_Embedding_Parameters.pattern_Path, "Pattern_Metadata_Dict.pickle").replace("\\", "/"), "wb") as f:
        pickle.dump(metadata_Dict, f, protocol=2);


if __name__ == '__main__':    
    Pattern_Generate_VCTK(
        wav_Path= "C:/Simulation_Raw_Data.Temp/VCTK/wav48"
        )
    Pattern_Generate_LS(
        data_Path= "C:/Simulation_Raw_Data.Temp/LibriSpeech/train"
        )
    Pattern_Generate_VC(   
        lexicon_Suffix= "2",
        wav_Path= "C:/Simulation_Raw_Data.Temp/VoxCeleb2/aac"
        )
    Pattern_Generate_VC(
        lexicon_Suffix= "1",
        wav_Path= "C:/Simulation_Raw_Data.Temp/VoxCeleb1/wav"
        )
    Metadata_Generate()