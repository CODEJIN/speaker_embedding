# Speaker embedding

This code is used to extract the speaker's feature from wav and make the speaker's embedding. The algorithm is based on the following papers:

    Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2017). Generalized end-to-end loss for speaker verification. arXiv preprint arXiv:1710.10467.
    Jia, Y., Zhang, Y., Weiss, R. J., Wang, Q., Shen, J., Ren, F., ... & Wu, Y. (2018). Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. arXiv preprint arXiv:1806.04558.
    
# Training and test

VCTK, LibriSpeech, VoxCeleb1, and VoxCeleb2 were used for model learning, and some of the test sets of VoxCeleb1 that were not learned were used for testing the learned model. Please refer to the following link for each dataset:

    VCTK: https://datashare.is.ed.ac.uk/handle/10283/2651
    LibriSpeech: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    VoxCeleb: http://www.openslr.org/12/

# t-SNE Result about the sentences of 10 non-trained talkers

<img src="https://user-images.githubusercontent.com/17133841/47399056-e2389480-d704-11e8-98fd-b973510b5e79.gif" width="300">  
<img src="https://user-images.githubusercontent.com/17133841/47399057-e2389480-d704-11e8-9fc4-581bf4d2887b.png" width="300">
