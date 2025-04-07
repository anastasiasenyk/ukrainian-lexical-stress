# ukrainian-lexical-stress


## For Nemo
```
# install dependencies
pip install "nemo_toolkit[all]"

cd ./accentor_nemo

# convert to Nemo format, manifest file (only if there is no train.json or eval.json)
python dataset_to_nemo_format.py

# taken from https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html#model-training-evaluation-and-inference

python g2p_train_and_evaluate.py \
    --config-path=$PWD \
    --config-name=g2p_t5 \
    model.train_ds.manifest_filepath=train.json \
    model.validation_ds.manifest_filepath=eval.json \
    model.test_ds.manifest_filepath=eval.json \
    trainer.devices=1 \
    do_training=True \
    do_testing=True
```