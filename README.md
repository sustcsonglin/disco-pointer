# disco-pointer
The official implementation of ACL2023: Don't Parse, Choose Spans! Continuous and Discontinuous Constituency Parsing via Autoregressive Span Selection

## Setup
Environment 
```
conda create -n parsing python=3.7
conda activate parsing
while read requirement; do pip install $requirement; done < requirement.txt 
```

Note: pytorch-lightning=1.2.4

Download preprocessed datasets from: [Google Drive](https://drive.google.com/drive/folders/1qFP2JbcltAJ-Jq3MpkS--0MGEIgyE6vQ?usp=sharing)

If you want to run discontinuous constituency parsing experiments, please make sure you have already installed [disco-dop](https://github.com/andreasvc/disco-dop)

# Run
Example:

python train.py +exp=tiger_bert_large seed=0,1,2 --multirun

### OOM
If you encounter out-of-memory issues, you could use gradient accumulation technique by changing the hyperparameters in configs/exp/*.yaml file and    letting accumulation * datamodule.max_token = 3000.  For example, set accumulation=15 and datamodule.max_token=200 if your GPU memory is small(e.g.,12GB). 


# Contact
Feel free to contact bestsonta@gmail.com if you have any questions.

# Credits
The code is based on [lightning+hydra](https://github.com/ashleve/lightning-hydra-template) template. I use [FastNLP](https://github.com/fastnlp/fastNLP) for loading data. I use lots of built-in modules (LSTMs, Biaffines, Triaffines, Dropout Layers, etc) from [Supar](https://github.com/yzhangcs/parser/tree/main/supar).  



