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

synthesizer_Parameters_Dict = {
    "mel_Dimension": 80,
    "batch_Size": 16,
    "learning_Rate": 0.002,
    "learning_Rate_Decay": "noam", #"none"
    "max_Queue": 40,
    "pattern_Sorting": True,
    "token_Embedding_Size": 512,
    "encoder_Conv_Filter_Count": 512,
    "encoder_Conv_Kernel_Size": 5,
    "encoder_Conv_Layer_Count": 3,
    "encoder_LSTM_Cell_Size": 256,
    "encoder_Zoneout_Rate": 0.1,
    "attention_Size": 128,
    "decoder_Prenet_Layer_Size": 256,
    "decoder_Prenet_Layer_Count": 2,
    "decoder_Prenet_Dropout_Rate": 0.5,    
    "decoder_LSTM_Cell_Size": 1024,
    "decoder_Zoneout_Rate": 0.1,
    "decoder_Output_Size_per_Step": 1,
    "decoder_Postnet_Conv_Filter_Count": 512,
    "decoder_Postnet_Conv_Kernal_Size": 5,
    "decoder_Postnet_Conv_Layer_Count": 5,
    "decoder_Max_Mel_Length": 1400,
    "pattern_Path": "D:/Simulation_Raw_Data/Multi_Speaker_TTS/Synthesizer",
    "extract_Path": "D:/Multi_Speaker_TTS_Result/Synthesizer",
    }

sound_Parameters = tf.contrib.training.HParams(**sound_Parameters_Dict);
speaker_Embedding_Parameters = tf.contrib.training.HParams(**speaker_Embedding_Parameters_Dict);
synthesizer_Parameters = tf.contrib.training.HParams(**synthesizer_Parameters_Dict);