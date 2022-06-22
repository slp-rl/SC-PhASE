# A Systematic Comparison of Phonetic Aware Techniques for Speech Enhancement  (PhaSE)  @interspeech2022
This is a PyTorch implementation of the pipeline presented in [A Systematic Comparison of Phonetic Aware Techniques for Speech Enhancement](put link) 
paper published @Interspeech2022.

**Abstract**<br> 
Speech enhancement has seen great improvement in recent
years using end-to-end neural networks. However, most models are agnostic to the spoken phonetic content. <br> 
Recently, several studies suggested phonetic-aware speech enhancement,
mostly using perceptual supervision. 
Yet, injecting phonetic features during model optimization can take additional forms
(e.g., model conditioning). <br> 
In this paper, we conduct a systematic comparison between different methods of incorporating phonetic information in a speech enhancement model.<br> 
By conducting a series of controlled experiments, we observe the
influence of different phonetic content models as well as various feature-injection techniques on enhancement performance,
considering both causal and non-causal models. <br> 
Specifically, we evaluate three settings for injecting phonetic information,
namely: i) feature conditioning; ii) perceptual supervision; and
iii) regularization. <br> 
Phonetic features are obtained using an intermediate layer of either a supervised pre-trained Automatic
Speech Recognition (ASR) model or by using a pre-trained
Self-Supervised Learning (SSL) model. <br> 
We further observe the
effect of choosing different embedding layers on performance,
considering both manual and learned configurations. 
Results suggest that using a SSL model as phonetic features outperforms the ASR one
in most cases. 
Interestingly, the conditioning setting performs best among the evaluated configurations.
[Paper](put link).

![Pipeline overview](img/Architechture.png)
---
### Basic Setup

#### Set virtual Env
do `pip install -r requirements.txt`.

Note: torch installation may depend on your cuda version. see [Install torch](https://pytorch.org/get-started/locally/)

#### Data preprocessing
1. Download all and unzip the [Valentini dataset](https://datashare.ed.ac.uk/handle/10283/2791) 
2. Down-sample each directory using `bash data_preprocessing_scripts/general/audio_resample_using_sox.sh <path to data dir> <path to target dir>`
3. Generate json files: `python data_preprocessing_scripts/speech_enhancement/valentini_egs_script.py --project_dir <full path to current project root> --dataset_base_dir <full path to the downsampled audio root, containing all downsampled dirs>`

#### Download pretrained HuBERT model
1. Download pretrained weights [link](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md)
2. Copy full path to the pretrained .pt file to `features_config.state_dict_path` field in configurations/main_config.yaml

---
### Train
Example run commands could be found at run_commands directory.