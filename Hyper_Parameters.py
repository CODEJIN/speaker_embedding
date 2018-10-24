import tensorflow as tf

sound_Parameters_Dict = {    
    "sample_Rate": 16000,    
    "spectrogram_Dimension": 1024,
    "mel_Frame_Shift": 12.5,
    "mel_Frame_Length": 50,
    }

speaker_Embedding_Parameters_Dict = {    
    "mel_Dimension": 40,
    "batch_Speaker": 64,
    "batch_Pattern_per_Speaker": 10,
    "embedding_Size": 256,
    "learning_Rate": 0.002,
    "pattern_Frame_Range": (140, 180),
    "loss_Method": "Softmax",   #"Contrast"
    "max_Queue": 100,
    "pattern_Path": "D:/Simulation_Raw_Data/Multi_Speaker_TTS/Speaker_Embedding",
    "extract_Path": "D:/Multi_Speaker_TTS_Result/Speaker_Embedding",
    }

sound_Parameters = tf.contrib.training.HParams(**sound_Parameters_Dict);
speaker_Embedding_Parameters = tf.contrib.training.HParams(**speaker_Embedding_Parameters_Dict);
