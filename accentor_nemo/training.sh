python g2p_train_and_evaluate.py \
    --config-path=$PWD \
    --config-name=g2p_t5 \
    model.train_ds.manifest_filepath=train.json \
    model.validation_ds.manifest_filepath=eval.json \
    model.test_ds.manifest_filepath=eval.json \
    trainer.devices=2 \
    do_training=True \
    do_testing=True