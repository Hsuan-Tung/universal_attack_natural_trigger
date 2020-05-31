# universal_attack_natural_trigger  
Universal Adversarial Attacks with Natural Triggers for Text Classification

## Dependencies  
Pytorch, AllenNLP, Hugging Face Transformers (see requirements.txt).

## Perform universal attacks  
First, download the pretrained ARAE model [here](https://drive.google.com/file/d/1h4GlTP1iVbQQQfZkSGoNtbcfCp1D2gzB/view?usp=sharing), and unzip into the "./ARAR/oneb_pretrained" folder.  

Then, go to sst or snli directory and run `python sst_attack.py` or `python snli_attack.py`.  
The argument `attack_class` is used to select the class label to attack, and the argument `len_lim` specifies the length of attack trigger.
