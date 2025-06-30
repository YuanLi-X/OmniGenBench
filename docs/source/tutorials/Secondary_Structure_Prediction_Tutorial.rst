RNA Secondary Structure Prediction
========================================================================

GitHub: https://github.com/yangheng95/OmniGenome OmniGenome Hub:
Huggingface Spaces

Introduction
------------

OmniGenome is a comprehensive package designed for pretrained genomic
foundation models (FMs) development and FM benchmark. OmniGenome have
the following key features: - Automated genomic FM benchmarking on
public genomic datasets - Scalable genomic FM training and fine-tuning
on genomic tasks - Diversified genomic FMs implementation - Easy-to-use
pipeline for genomic FM development with no coding expertise required -
Accessible OmniGenome Hub for sharing FMs, datasets, and pipelines -
Extensive documentation and tutorials for genomic FM development

We begin to introduce OmniGenome by delivering a demonstration to train
a model to predict RNA secondary structures. The dataset used in this
demonstration is the bpRNA dataset which contains RNA sequences and
their corresponding secondary structures. The secondary structure of an
RNA sequence is a set of base pairs that describe the folding of the RNA
molecule. The secondary structure of an RNA sequence is important for
understanding the function of the RNA molecule. In this demonstration,
we will train a model to predict the secondary structure of an RNA
sequence given its primary sequence.

Requirements
------------

OmniGenome requires the following recommended dependencies: - Python
3.9+ - PyTorch 2.0.0+ - Transformers 4.37.0+ - Pandas 1.3.3+ - Others in
case of specific tasks

pip install OmniGenome

Fine-tuning Genomic FMs for RNA Secondary Structure Prediction
--------------------------------------------------------------

Step 1: Import Libraries
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    
    import autocuda
    import torch
    from metric_visualizer import MetricVisualizer
    
    from omnigenome import OmniGenomeDatasetForTokenClassification
    from omnigenome import ClassificationMetric
    from omnigenome import OmniSingleNucleotideTokenizer, OmniKmersTokenizer
    from omnigenome import OmniGenomeModelForTokenClassification
    from omnigenome import Trainer

Step 2: Define and Initialize the Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Predefined dataset label mapping
    label2id = {"(": 0, ")": 1, ".": 2}
    
    # The is FM is exclusively powered by the OmniGenome package
    model_name_or_path = "anonymous8/OmniGenome-186M"
    
    # Generally, we use the tokenizers from transformers library, such as AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # However, OmniGenome provides specialized tokenizers for genomic data, such as single nucleotide tokenizer and k-mers tokenizer
    # we can force the tokenizer to be used in the model
    tokenizer = OmniSingleNucleotideTokenizer.from_pretrained(model_name_or_path)

Step 3: Define and Initialize the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # We have implemented a diverse set of genomic models in OmniGenome, please refer to the documentation for more details
    ssp_model = OmniGenomeModelForTokenClassification(
        model_name_or_path,
        tokenizer=tokenizer,
        label2id=label2id,
    )


.. parsed-literal::

    You are using a model of type omnigenome to instantiate a model of type mprna. This is not supported for all configurations of models and can yield errors.
    Some weights of the model checkpoint at anonymous8/OmniGenome-186M were not used when initializing OmniGenomeModel: ['classifier.bias', 'classifier.weight', 'dense.bias', 'dense.weight', 'lm_head.bias', 'lm_head.decoder.weight', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
    - This IS expected if you are initializing OmniGenomeModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing OmniGenomeModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-186M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2024-08-11 17:43:16] (0.0.8alpha) Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.0.8alpha', 'torch_version': '2.1.2+cu12.1+gita8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb', 'transformers_version': '4.43.2', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'OmniSingleNucleotideTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-186M
    Model Type: omnigenome
    Model Architecture: ['OmniGenomeModel', 'OmniGenomeForTokenClassification', 'OmniGenomeForMaskedLM', 'OmniGenomeModelForSeq2SeqLM', 'OmniGenomeForTSequenceClassification', 'OmniGenomeForTokenClassification', 'OmniGenomeForSeq2SeqLM']
    Model Parameters: 185.886801 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-186M",
      "architectures": [
        "OmniGenomeModel",
        "OmniGenomeForTokenClassification",
        "OmniGenomeForMaskedLM",
        "OmniGenomeModelForSeq2SeqLM",
        "OmniGenomeForTSequenceClassification",
        "OmniGenomeForTokenClassification",
        "OmniGenomeForSeq2SeqLM"
      ],
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-186M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 720,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2560,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "mprna",
      "num_attention_heads": 30,
      "num_generation": 50,
      "num_hidden_layers": 32,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.43.2",
      "use_cache": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    

Step 4: Define and Load the Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # necessary hyperparameters
    epochs = 10
    learning_rate = 2e-5
    weight_decay = 1e-5
    batch_size = 8
    max_length = 512
    seeds = [45]  # Each seed will be used for one run
    
    
    # Load the dataset according to the path
    train_file = "toy_datasets/Archive2/train.json"
    test_file = "toy_datasets/Archive2/test.json"
    valid_file = "toy_datasets/Archive2/valid.json"
    
    train_set = OmniGenomeDatasetForTokenClassification(
        data_source=train_file,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    test_set = OmniGenomeDatasetForTokenClassification(
        data_source=test_file,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    valid_set = OmniGenomeDatasetForTokenClassification(
        data_source=valid_file,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


.. parsed-literal::

    [2024-08-11 17:43:16] (0.0.8alpha) Detected max_length=512 in the dataset, using it as the max_length.
    [2024-08-11 17:43:16] (0.0.8alpha) Loading data from toy_datasets/Archive2/train.json...
    [2024-08-11 17:43:16] (0.0.8alpha) Loaded 608 examples from toy_datasets/Archive2/train.json
    [2024-08-11 17:43:16] (0.0.8alpha) Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████████████████████████████████████████████████████████████████████████| 608/608 [00:00<00:00, 5249.80it/s]
    

.. parsed-literal::

    [2024-08-11 17:43:17] (0.0.8alpha) {'avg_seq_len': 130.54276315789474, 'max_seq_len': 501, 'min_seq_len': 56, 'avg_label_len': 501.0, 'max_label_len': 501, 'min_label_len': 501}
    [2024-08-11 17:43:17] (0.0.8alpha) Detected max_length=512 in the dataset, using it as the max_length.
    [2024-08-11 17:43:17] (0.0.8alpha) Loading data from toy_datasets/Archive2/test.json...
    [2024-08-11 17:43:17] (0.0.8alpha) Loaded 82 examples from toy_datasets/Archive2/test.json
    [2024-08-11 17:43:17] (0.0.8alpha) Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|████████████████████████████████████████████████████████████████████████████████| 82/82 [00:00<00:00, 3625.84it/s]
    

.. parsed-literal::

    [2024-08-11 17:43:17] (0.0.8alpha) {'avg_seq_len': 131.23170731707316, 'max_seq_len': 321, 'min_seq_len': 67, 'avg_label_len': 321.0, 'max_label_len': 321, 'min_label_len': 321}
    [2024-08-11 17:43:17] (0.0.8alpha) Detected max_length=512 in the dataset, using it as the max_length.
    [2024-08-11 17:43:17] (0.0.8alpha) Loading data from toy_datasets/Archive2/valid.json...
    [2024-08-11 17:43:17] (0.0.8alpha) Loaded 76 examples from toy_datasets/Archive2/valid.json
    [2024-08-11 17:43:17] (0.0.8alpha) Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|████████████████████████████████████████████████████████████████████████████████| 76/76 [00:00<00:00, 5782.41it/s]

.. parsed-literal::

    [2024-08-11 17:43:17] (0.0.8alpha) {'avg_seq_len': 117.39473684210526, 'max_seq_len': 308, 'min_seq_len': 60, 'avg_label_len': 308.0, 'max_label_len': 308, 'min_label_len': 308}
    

.. parsed-literal::

    
    

Step 5: Define the Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

We have implemented a diverse set of genomic metrics in OmniGenome,
please refer to the documentation for more details. Users can also
define their own metrics by inheriting the ``OmniGenomeMetric`` class.
The ``compute_metrics`` can be a metric function list and each metric
function should return a dictionary of metrics.

.. code:: ipython3

    compute_metrics = [
        ClassificationMetric(ignore_y=-100).accuracy_score,
        ClassificationMetric(ignore_y=-100, average="macro").f1_score,
        ClassificationMetric(ignore_y=-100).matthews_corrcoef,
    ]
    

Step 6: Define and Initialize the Trainer
-----------------------------------------

.. code:: ipython3

    # Initialize the MetricVisualizer for logging the metrics
    mv = MetricVisualizer(name="OmniGenome-186M-SSP")
    
    for seed in seeds:
        optimizer = torch.optim.AdamW(
            ssp_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        trainer = Trainer(
            model=ssp_model,
            train_loader=train_loader,
            eval_loader=valid_loader,
            test_loader=test_loader,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            seeds=seed,
            device=autocuda.auto_cuda(),
        )
    
        metrics = trainer.train()
        test_metrics = metrics["test"][-1]
        mv.log(model_name_or_path.split("/")[-1], "F1", test_metrics["f1_score"])
        mv.log(
            model_name_or_path.split("/")[-1],
            "Accuracy",
            test_metrics["accuracy_score"],
        )
        print(metrics)
        mv.summary()


.. parsed-literal::

    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  7.43it/s]
    

.. parsed-literal::

    [2024-08-11 17:43:19] (0.0.8alpha) {'accuracy_score': 0.2790193842645382, 'f1_score': 0.28151975296578563, 'matthews_corrcoef': -0.09291127922709266}
    

.. parsed-literal::

    Epoch 1/10 Loss: 0.7989: 100%|█████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.54it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  7.99it/s]
    

.. parsed-literal::

    [2024-08-11 17:44:11] (0.0.8alpha) {'accuracy_score': 0.8913340935005701, 'f1_score': 0.8935400779001638, 'matthews_corrcoef': 0.8353253240117546}
    

.. parsed-literal::

    Epoch 2/10 Loss: 0.6545: 100%|█████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.54it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.16it/s]
    

.. parsed-literal::

    [2024-08-11 17:45:02] (0.0.8alpha) {'accuracy_score': 0.9076396807297605, 'f1_score': 0.9095038559875431, 'matthews_corrcoef': 0.8604032983011348}
    

.. parsed-literal::

    Epoch 3/10 Loss: 0.6302: 100%|█████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.18it/s]
    

.. parsed-literal::

    [2024-08-11 17:45:54] (0.0.8alpha) {'accuracy_score': 0.9148232611174458, 'f1_score': 0.9163503175903402, 'matthews_corrcoef': 0.86969111358666}
    

.. parsed-literal::

    Epoch 4/10 Loss: 0.6151: 100%|█████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.19it/s]
    

.. parsed-literal::

    [2024-08-11 17:46:45] (0.0.8alpha) {'accuracy_score': 0.9169897377423033, 'f1_score': 0.9185686268915924, 'matthews_corrcoef': 0.8725737867525207}
    

.. parsed-literal::

    Epoch 5/10 Loss: 0.6071: 100%|█████████████████████████████████████████████████████████| 76/76 [00:48<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.22it/s]
    

.. parsed-literal::

    [2024-08-11 17:47:36] (0.0.8alpha) {'accuracy_score': 0.9189281641961231, 'f1_score': 0.9205276415383489, 'matthews_corrcoef': 0.875436812852734}
    

.. parsed-literal::

    Epoch 6/10 Loss: 0.6013: 100%|█████████████████████████████████████████████████████████| 76/76 [00:48<00:00,  1.56it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.21it/s]
    

.. parsed-literal::

    [2024-08-11 17:48:28] (0.0.8alpha) {'accuracy_score': 0.9210946408209806, 'f1_score': 0.9226092911100953, 'matthews_corrcoef': 0.879263171602823}
    

.. parsed-literal::

    Epoch 7/10 Loss: 0.5989: 100%|█████████████████████████████████████████████████████████| 76/76 [00:48<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.24it/s]
    

.. parsed-literal::

    [2024-08-11 17:49:19] (0.0.8alpha) {'accuracy_score': 0.9238312428734321, 'f1_score': 0.9253576750498466, 'matthews_corrcoef': 0.8831977559814651}
    

.. parsed-literal::

    Epoch 8/10 Loss: 0.5979: 100%|█████████████████████████████████████████████████████████| 76/76 [00:48<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.23it/s]
    

.. parsed-literal::

    [2024-08-11 17:50:10] (0.0.8alpha) {'accuracy_score': 0.9234891676168757, 'f1_score': 0.9250099970359921, 'matthews_corrcoef': 0.8820785908253933}
    

.. parsed-literal::

    Epoch 9/10 Loss: 0.5955: 100%|█████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.17it/s]
    

.. parsed-literal::

    [2024-08-11 17:51:00] (0.0.8alpha) {'accuracy_score': 0.9240592930444698, 'f1_score': 0.9255602479349917, 'matthews_corrcoef': 0.883211983456326}
    

.. parsed-literal::

    Epoch 10/10 Loss: 0.5913: 100%|████████████████████████████████████████████████████████| 76/76 [00:49<00:00,  1.55it/s]
    Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.24it/s]
    

.. parsed-literal::

    [2024-08-11 17:51:51] (0.0.8alpha) {'accuracy_score': 0.9225769669327252, 'f1_score': 0.9241115922227455, 'matthews_corrcoef': 0.8821062314790764}
    

.. parsed-literal::

    Testing: 100%|█████████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00,  8.05it/s]
    

.. parsed-literal::

    [2024-08-11 17:51:53] (0.0.8alpha) {'accuracy_score': 0.902897046333868, 'f1_score': 0.9044334792769698, 'matthews_corrcoef': 0.8503789642989459}
    {'valid': [{'accuracy_score': 0.2790193842645382, 'f1_score': 0.28151975296578563, 'matthews_corrcoef': -0.09291127922709266}, {'accuracy_score': 0.8913340935005701, 'f1_score': 0.8935400779001638, 'matthews_corrcoef': 0.8353253240117546}], 'best_valid': {'accuracy_score': 0.9240592930444698, 'f1_score': 0.9255602479349917, 'matthews_corrcoef': 0.883211983456326}, 'test': [{'accuracy_score': 0.902897046333868, 'f1_score': 0.9044334792769698, 'matthews_corrcoef': 0.8503789642989459}]}
    
    ----------------------------------------------- Raw Metric Records -----------------------------------------------
    ╒══════════╤═════════════════╤══════════════════════╤═══════════╤══════════╤═══════╤═══════╤══════════╤══════════╕
    │ Metric   │ Trial           │ Values               │  Average  │  Median  │  Std  │  IQR  │   Min    │   Max    │
    ╞══════════╪═════════════════╪══════════════════════╪═══════════╪══════════╪═══════╪═══════╪══════════╪══════════╡
    │ F1       │ OmniGenome-186M │ [0.9044334792769698] │ 0.904433  │ 0.904433 │   0   │   0   │ 0.904433 │ 0.904433 │
    ├──────────┼─────────────────┼──────────────────────┼───────────┼──────────┼───────┼───────┼──────────┼──────────┤
    │ Accuracy │ OmniGenome-186M │ [0.902897046333868]  │ 0.902897  │ 0.902897 │   0   │   0   │ 0.902897 │ 0.902897 │
    ╘══════════╧═════════════════╧══════════════════════╧═══════════╧══════════╧═══════╧═══════╧══════════╧══════════╛
    -------------------------------- https://github.com/yangheng95/metric_visualizer --------------------------------
    
    

.. parsed-literal::

    C:\Users\chuan\miniconda3\lib\site-packages\metric_visualizer\utils.py:31: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
      self.skewness = stats.skew(self.data, keepdims=True)
    

Step 7. Experimental Results Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The experimental results are visualized in the following plots. The
plots show the F1 score and accuracy of the model on the test set for
each run. The average F1 score and accuracy are also shown.

\|### Step 8. Model Checkpoint for Sharing The model checkpoint can be
saved and shared with others for further use. The model checkpoint can
be loaded using the following code:

**Regular checkpointing and resuming are good practices to save the
model at different stages of training.**

.. code:: ipython3

    path_to_save = "OmniGenome-186M-SSP"
    ssp_model.save(path_to_save, overwrite=True)
    
    # Load the model checkpoint
    ssp_model = ssp_model.load(path_to_save)
    results = ssp_model.inference("CAGUGCCGAGGCCACGCGGAGAACGAUCGAGGGUACAGCACUA")
    print(results["predictions"])
    print("logits:", results["logits"])


.. parsed-literal::

    [2024-08-11 17:51:55] (0.0.8alpha) The model is saved to OmniGenome-186M-SSP.
    [2024-08-11 17:51:58] (0.0.8alpha) Warning: The value of the key torch_dtype in the loaded model is torch.float16, but the current value is float16.
    [2024-08-11 17:51:58] (0.0.8alpha) Warning: The value of the key _name_or_path in the loaded model is OmniGenome-186M-SSP, but the current value is anonymous8/OmniGenome-186M.
    [2024-08-11 17:51:58] (0.0.8alpha) Warning: The value of the key _commit_hash in the loaded model is None, but the current value is 0ea2f7c3929aa2d3a2b004fad73ae16afe17d18a.
    [2024-08-11 17:51:58] (0.0.8alpha) Warning: The value of the key transformers_version in the loaded model is 4.43.2, but the current value is 4.41.0.dev0.
    [2024-08-11 17:51:58] (0.0.8alpha) Warning: The value of the key model_type in the loaded model is mprna, but the current value is omnigenome.
    ['.', '(', '(', '(', '(', '(', '.', '.', '.', '.', '(', '(', '(', '.', '(', '.', '(', '(', '(', '.', '.', '.', '.', '.', '.', '.', ')', ')', ')', '.', ')', '.', ')', '.', '.', '.', '.', ')', ')', ')', ')', ')', '.']
    logits: tensor([[8.0241e-04, 6.8535e-04, 9.9851e-01],
            [1.8072e-03, 2.7458e-04, 9.9792e-01],
            [9.9968e-01, 1.4969e-04, 1.7153e-04],
            [9.9977e-01, 1.2595e-04, 1.0330e-04],
            [9.9973e-01, 1.5334e-04, 1.1417e-04],
            [9.9977e-01, 1.1016e-04, 1.1670e-04],
            [9.9974e-01, 1.4174e-04, 1.1885e-04],
            [1.6035e-04, 8.9402e-05, 9.9975e-01],
            [1.2057e-04, 1.2549e-04, 9.9975e-01],
            [1.0425e-04, 1.2844e-04, 9.9977e-01],
            [1.0099e-04, 1.1066e-04, 9.9979e-01],
            [9.9936e-01, 2.3561e-04, 4.0091e-04],
            [9.9964e-01, 1.5549e-04, 2.0940e-04],
            [9.9949e-01, 1.4136e-04, 3.7019e-04],
            [3.0048e-04, 1.4218e-04, 9.9956e-01],
            [9.9924e-01, 2.6267e-04, 4.9686e-04],
            [2.3464e-01, 1.5779e-03, 7.6379e-01],
            [9.9944e-01, 2.1302e-04, 3.4890e-04],
            [9.9943e-01, 2.3252e-04, 3.3931e-04],
            [9.8945e-01, 4.7213e-04, 1.0077e-02],
            [1.9276e-04, 1.0888e-04, 9.9970e-01],
            [1.8010e-04, 1.4927e-04, 9.9967e-01],
            [8.2282e-05, 1.0816e-04, 9.9981e-01],
            [8.3079e-05, 1.3431e-04, 9.9978e-01],
            [9.0559e-05, 2.3991e-04, 9.9967e-01],
            [8.2159e-05, 1.9781e-04, 9.9972e-01],
            [1.2829e-04, 1.6677e-04, 9.9970e-01],
            [2.1760e-03, 6.7431e-01, 3.2351e-01],
            [2.3911e-04, 9.9938e-01, 3.8139e-04],
            [1.3654e-04, 9.9912e-01, 7.4456e-04],
            [2.3898e-04, 2.9068e-04, 9.9947e-01],
            [7.0051e-04, 6.3645e-01, 3.6285e-01],
            [3.5826e-04, 7.7297e-03, 9.9191e-01],
            [6.9229e-04, 7.6308e-01, 2.3623e-01],
            [3.1615e-04, 3.1499e-02, 9.6818e-01],
            [6.1732e-05, 1.1555e-04, 9.9982e-01],
            [6.4652e-05, 2.9019e-04, 9.9965e-01],
            [6.6991e-05, 1.6510e-04, 9.9977e-01],
            [9.8428e-05, 9.9977e-01, 1.3399e-04],
            [1.2540e-04, 9.9977e-01, 1.0871e-04],
            [1.0226e-04, 9.9977e-01, 1.2271e-04],
            [1.1149e-04, 9.9978e-01, 1.0948e-04],
            [1.0455e-04, 9.9966e-01, 2.3631e-04],
            [1.1773e-04, 1.8630e-04, 9.9970e-01],
            [5.4235e-04, 1.9674e-03, 9.9749e-01]], device='cuda:0')
    

.. parsed-literal::

    C:\Users\chuan\miniconda3\lib\site-packages\transformers\tokenization_utils_base.py:2906: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
      warnings.warn(
    

What if someone doesn’t know how to initialize the model?
=========================================================

.. code:: ipython3

    # We can load the model checkpoint using the ModelHub
    from omnigenome import ModelHub
    
    ssp_model = ModelHub.load("OmniGenome-186M-SSP")
    results = ssp_model.inference("CAGUGCCGAGGCCACGCGGAGAACGAUCGAGGGUACAGCACUA")
    print(results["predictions"])
    print("logits:", results["logits"])


.. parsed-literal::

    [2024-08-11 17:52:00] (0.0.8alpha) Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.0.8alpha', 'torch_version': '2.1.2+cu12.1+gita8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb', 'transformers_version': '4.43.2', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'OmniSingleNucleotideTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: OmniGenome-186M-SSP
    Model Type: mprna
    Model Architecture: ['OmniGenomeModel']
    Model Parameters: 185.886801 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "OmniGenome-186M-SSP",
      "architectures": [
        "OmniGenomeModel"
      ],
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-186M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 720,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2560,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "metadata": {
        "library_name": "OmniGenome",
        "model_cls": "OmniGenomeModelForTokenClassification",
        "model_name": "OmniGenomeModelForTokenClassification",
        "omnigenome_version": "0.0.8alpha",
        "tokenizer_cls": "OmniSingleNucleotideTokenizer",
        "torch_version": "2.1.2+cu12.1+gita8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb",
        "transformers_version": "4.43.2"
      },
      "model_type": "mprna",
      "num_attention_heads": 30,
      "num_generation": 50,
      "num_hidden_layers": 32,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float16",
      "transformers_version": "4.43.2",
      "use_cache": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    ['.', '(', '(', '(', '(', '(', '.', '.', '.', '.', '(', '(', '(', '.', '(', '.', '(', '(', '(', '.', '.', '.', '.', '.', '.', '.', ')', ')', ')', '.', ')', '.', ')', '.', '.', '.', '.', ')', ')', ')', ')', ')', '.']
    logits: tensor([[8.0241e-04, 6.8535e-04, 9.9851e-01],
            [1.8072e-03, 2.7458e-04, 9.9792e-01],
            [9.9968e-01, 1.4969e-04, 1.7153e-04],
            [9.9977e-01, 1.2595e-04, 1.0330e-04],
            [9.9973e-01, 1.5334e-04, 1.1417e-04],
            [9.9977e-01, 1.1016e-04, 1.1670e-04],
            [9.9974e-01, 1.4174e-04, 1.1885e-04],
            [1.6035e-04, 8.9402e-05, 9.9975e-01],
            [1.2057e-04, 1.2549e-04, 9.9975e-01],
            [1.0425e-04, 1.2844e-04, 9.9977e-01],
            [1.0099e-04, 1.1066e-04, 9.9979e-01],
            [9.9936e-01, 2.3561e-04, 4.0091e-04],
            [9.9964e-01, 1.5549e-04, 2.0940e-04],
            [9.9949e-01, 1.4136e-04, 3.7019e-04],
            [3.0048e-04, 1.4218e-04, 9.9956e-01],
            [9.9924e-01, 2.6267e-04, 4.9686e-04],
            [2.3464e-01, 1.5779e-03, 7.6379e-01],
            [9.9944e-01, 2.1302e-04, 3.4890e-04],
            [9.9943e-01, 2.3252e-04, 3.3931e-04],
            [9.8945e-01, 4.7213e-04, 1.0077e-02],
            [1.9276e-04, 1.0888e-04, 9.9970e-01],
            [1.8010e-04, 1.4927e-04, 9.9967e-01],
            [8.2282e-05, 1.0816e-04, 9.9981e-01],
            [8.3079e-05, 1.3431e-04, 9.9978e-01],
            [9.0559e-05, 2.3991e-04, 9.9967e-01],
            [8.2159e-05, 1.9781e-04, 9.9972e-01],
            [1.2829e-04, 1.6677e-04, 9.9970e-01],
            [2.1760e-03, 6.7431e-01, 3.2351e-01],
            [2.3911e-04, 9.9938e-01, 3.8139e-04],
            [1.3654e-04, 9.9912e-01, 7.4456e-04],
            [2.3898e-04, 2.9068e-04, 9.9947e-01],
            [7.0051e-04, 6.3645e-01, 3.6285e-01],
            [3.5826e-04, 7.7297e-03, 9.9191e-01],
            [6.9229e-04, 7.6308e-01, 2.3623e-01],
            [3.1615e-04, 3.1499e-02, 9.6818e-01],
            [6.1732e-05, 1.1555e-04, 9.9982e-01],
            [6.4652e-05, 2.9019e-04, 9.9965e-01],
            [6.6991e-05, 1.6510e-04, 9.9977e-01],
            [9.8428e-05, 9.9977e-01, 1.3399e-04],
            [1.2540e-04, 9.9977e-01, 1.0871e-04],
            [1.0226e-04, 9.9977e-01, 1.2271e-04],
            [1.1149e-04, 9.9978e-01, 1.0948e-04],
            [1.0455e-04, 9.9966e-01, 2.3631e-04],
            [1.1773e-04, 1.8630e-04, 9.9970e-01],
            [5.4235e-04, 1.9674e-03, 9.9749e-01]], device='cuda:0')
    

Step 8. Model Inference
-----------------------

.. code:: ipython3

    examples = [
        "GCUGGGAUGUUGGCUUAGAAGCAGCCAUCAUUUAAAGAGUGCGUAACAGCUCACCAGC",
        "AUCUGUACUAGUUAGCUAACUAGAUCUGUAUCUGGCGGUUCCGUGGAAGAACUGACGUGUUCAUAUUCCCGACCGCAGCCCUGGGAGACGUCUCAGAGGC",
    ]
    
    results = ssp_model.inference(examples)
    structures = ["".join(prediction) for prediction in results["predictions"]]
    print(results)
    print(structures)


.. parsed-literal::

    {'predictions': [['(', '(', '(', '(', '(', '.', '(', '(', '(', '.', '(', '(', '(', '(', '(', '.', '.', '.', '.', '.', '.', '.', '.', ')', ')', ')', ')', '.', ')', ')', ')', '.', '.', '.', '.', '.', '(', '(', '(', '(', '.', '.', '.', '.', '.', '.', '.', '.', ')', ')', ')', ')', '.', ')', ')', ')', ')', ')'], ['.', '.', '.', '.', '.', '.', '.', '(', '(', '(', '(', '(', '.', '.', '.', '.', '.', '.', ')', ')', ')', ')', ')', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '(', '(', '(', '(', '.', '.', '.', '.', '.', '.', '(', '(', '(', '.', '.', '(', '(', '(', '.', '.', '.', '.', '.', '.', '.', ')', ')', ')', '.', '.', '.', ')', ')', ')', '.', '.', ')', ')', ')', ')', ')', ')', '.', '(', '(', '(', '(', '(', '(', '(', '(', '(', '.', '.', '.', '.', ')', ')', ')', ')', ')', ')', '.', ')', ')', ')']], 'logits': tensor([[[2.4458e-04, 2.9453e-04, 9.9946e-01],
             [9.9968e-01, 1.8715e-04, 1.3058e-04],
             [9.9971e-01, 1.7857e-04, 1.1603e-04],
             [9.9969e-01, 1.9235e-04, 1.2017e-04],
             [9.9974e-01, 1.6477e-04, 9.5490e-05],
             [9.9973e-01, 1.7072e-04, 1.0310e-04],
             [3.8932e-02, 4.0654e-04, 9.6066e-01],
             [9.9081e-01, 2.5896e-04, 8.9282e-03],
             [9.9954e-01, 7.5703e-05, 3.8098e-04],
             [9.9973e-01, 6.5632e-05, 2.0909e-04],
             [2.3895e-04, 7.7541e-05, 9.9968e-01],
             [9.9162e-01, 3.2099e-04, 8.0553e-03],
             [9.9925e-01, 1.8155e-04, 5.6669e-04],
             [9.9848e-01, 2.5072e-04, 1.2702e-03],
             [9.9588e-01, 2.8798e-04, 3.8292e-03],
             [9.7094e-01, 3.9280e-04, 2.8666e-02],
             [1.2957e-04, 6.7293e-05, 9.9980e-01],
             [9.2345e-05, 6.8185e-05, 9.9984e-01],
             [9.1842e-05, 7.8858e-05, 9.9983e-01],
             [6.2752e-05, 8.0315e-05, 9.9986e-01],
             [6.7975e-05, 8.9869e-05, 9.9984e-01],
             [1.0997e-04, 2.0999e-04, 9.9968e-01],
             [6.6543e-05, 1.7473e-04, 9.9976e-01],
             [5.0369e-04, 1.6350e-01, 8.3600e-01],
             [9.0685e-05, 9.9954e-01, 3.6993e-04],
             [1.0369e-04, 9.9932e-01, 5.7892e-04],
             [6.1102e-05, 9.9971e-01, 2.2508e-04],
             [6.7252e-05, 9.9913e-01, 8.0190e-04],
             [1.0101e-04, 1.5238e-04, 9.9975e-01],
             [9.6041e-05, 9.9968e-01, 2.2805e-04],
             [4.0995e-04, 8.3305e-01, 1.6654e-01],
             [3.2397e-04, 8.9397e-01, 1.0571e-01],
             [1.4482e-04, 1.3955e-02, 9.8590e-01],
             [6.9360e-05, 1.0573e-04, 9.9982e-01],
             [9.7087e-05, 1.0292e-04, 9.9980e-01],
             [8.7814e-05, 1.0701e-04, 9.9981e-01],
             [1.1646e-04, 1.1553e-04, 9.9977e-01],
             [9.9965e-01, 1.4555e-04, 2.0718e-04],
             [9.9954e-01, 2.0246e-04, 2.6042e-04],
             [9.9966e-01, 2.0734e-04, 1.3327e-04],
             [9.9941e-01, 2.7658e-04, 3.1286e-04],
             [1.7521e-04, 1.8747e-04, 9.9964e-01],
             [1.4355e-04, 9.9796e-05, 9.9976e-01],
             [9.4866e-05, 9.8983e-05, 9.9981e-01],
             [6.6438e-05, 8.6681e-05, 9.9985e-01],
             [5.4376e-05, 7.8099e-05, 9.9987e-01],
             [5.3411e-05, 1.1499e-04, 9.9983e-01],
             [6.6659e-05, 2.9356e-04, 9.9964e-01],
             [6.7383e-05, 2.9874e-04, 9.9963e-01],
             [1.1811e-04, 9.9960e-01, 2.7738e-04],
             [1.3575e-04, 9.9971e-01, 1.5813e-04],
             [7.3413e-05, 9.9977e-01, 1.5353e-04],
             [7.1526e-05, 9.9979e-01, 1.3900e-04],
             [7.8159e-05, 3.5420e-04, 9.9957e-01],
             [2.0607e-04, 9.9963e-01, 1.6156e-04],
             [2.3425e-04, 9.9959e-01, 1.7652e-04],
             [2.2851e-04, 9.9949e-01, 2.7702e-04],
             [2.0683e-04, 9.9956e-01, 2.3107e-04],
             [3.4234e-04, 9.9909e-01, 5.6296e-04],
             [2.3100e-04, 6.6516e-04, 9.9910e-01],
             [8.5801e-05, 1.1262e-04, 9.9980e-01],
             [1.0100e-04, 1.4332e-04, 9.9976e-01],
             [1.2906e-04, 1.4361e-04, 9.9973e-01],
             [1.7723e-04, 1.5324e-04, 9.9967e-01],
             [2.0009e-04, 1.3927e-04, 9.9966e-01],
             [4.3333e-04, 1.4627e-04, 9.9942e-01],
             [1.5578e-04, 6.9429e-05, 9.9977e-01],
             [1.3025e-04, 8.9878e-05, 9.9978e-01],
             [1.5267e-04, 8.9521e-05, 9.9976e-01],
             [2.2551e-04, 9.4320e-05, 9.9968e-01],
             [9.5946e-05, 7.5395e-05, 9.9983e-01],
             [7.3750e-05, 6.2262e-05, 9.9986e-01],
             [6.7474e-05, 6.6945e-05, 9.9987e-01],
             [7.9817e-05, 7.6630e-05, 9.9984e-01],
             [8.7196e-05, 1.0430e-04, 9.9981e-01],
             [8.1214e-05, 1.0127e-04, 9.9982e-01],
             [6.2139e-05, 8.7996e-05, 9.9985e-01],
             [7.6052e-05, 1.5072e-04, 9.9977e-01],
             [1.3132e-04, 2.3917e-04, 9.9963e-01],
             [1.2195e-04, 4.7527e-04, 9.9940e-01],
             [7.3472e-04, 1.6395e-01, 8.3531e-01],
             [3.2393e-04, 9.9711e-01, 2.5650e-03],
             [6.9540e-04, 8.5083e-01, 1.4847e-01],
             [1.0265e-04, 3.9632e-04, 9.9950e-01],
             [9.1168e-05, 1.1855e-04, 9.9979e-01],
             [1.0727e-04, 1.2373e-04, 9.9977e-01],
             [1.5165e-04, 1.4414e-04, 9.9970e-01],
             [1.1259e-04, 1.6083e-04, 9.9973e-01],
             [1.1293e-04, 1.3731e-04, 9.9975e-01],
             [1.3517e-04, 1.1040e-04, 9.9975e-01],
             [1.6120e-04, 1.1981e-04, 9.9972e-01],
             [2.0792e-04, 1.5390e-04, 9.9964e-01],
             [3.7049e-04, 2.6302e-04, 9.9937e-01],
             [5.7098e-04, 3.1766e-04, 9.9911e-01],
             [2.3590e-03, 7.7065e-04, 9.9687e-01],
             [4.7256e-03, 9.3478e-04, 9.9434e-01],
             [4.9784e-02, 3.7355e-03, 9.4648e-01],
             [1.4179e-04, 7.7191e-05, 9.9978e-01],
             [1.5795e-04, 9.7193e-05, 9.9974e-01],
             [1.1839e-04, 8.7829e-05, 9.9979e-01],
             [1.0780e-04, 8.3822e-05, 9.9981e-01],
             [8.9514e-05, 8.0147e-05, 9.9983e-01]],
    
            [[3.9569e-04, 2.4930e-04, 9.9936e-01],
             [1.9868e-04, 1.1945e-04, 9.9968e-01],
             [5.0411e-04, 1.7495e-04, 9.9932e-01],
             [2.8695e-04, 1.4937e-04, 9.9956e-01],
             [4.5617e-04, 1.7753e-04, 9.9937e-01],
             [6.7205e-04, 2.6345e-04, 9.9906e-01],
             [2.3186e-04, 1.2470e-04, 9.9964e-01],
             [1.3745e-04, 8.0860e-05, 9.9978e-01],
             [9.9966e-01, 1.5046e-04, 1.8733e-04],
             [9.9973e-01, 1.1524e-04, 1.5680e-04],
             [9.9969e-01, 1.0497e-04, 2.0725e-04],
             [9.9972e-01, 1.2338e-04, 1.5183e-04],
             [9.9948e-01, 1.4946e-04, 3.7327e-04],
             [1.6641e-04, 6.9403e-05, 9.9976e-01],
             [9.3493e-05, 7.4080e-05, 9.9983e-01],
             [1.0837e-04, 8.3742e-05, 9.9981e-01],
             [1.0277e-04, 7.7325e-05, 9.9982e-01],
             [6.8879e-05, 9.2819e-05, 9.9984e-01],
             [6.4741e-05, 1.0482e-04, 9.9983e-01],
             [1.2431e-04, 9.9643e-01, 3.4445e-03],
             [1.6579e-04, 9.9967e-01, 1.6150e-04],
             [3.3586e-04, 9.9947e-01, 1.9213e-04],
             [2.2018e-04, 9.9954e-01, 2.4348e-04],
             [1.8809e-04, 9.9962e-01, 1.9520e-04],
             [1.2736e-04, 9.5000e-05, 9.9978e-01],
             [1.0255e-04, 8.5937e-05, 9.9981e-01],
             [7.6329e-05, 8.9930e-05, 9.9983e-01],
             [1.2604e-04, 9.4003e-05, 9.9978e-01],
             [9.9790e-05, 1.0354e-04, 9.9980e-01],
             [1.0511e-04, 8.9037e-05, 9.9981e-01],
             [8.6558e-05, 7.4615e-05, 9.9984e-01],
             [1.4547e-04, 1.4036e-04, 9.9971e-01],
             [1.9739e-04, 1.4267e-04, 9.9966e-01],
             [4.2130e-04, 3.0790e-04, 9.9927e-01],
             [3.5211e-04, 1.4888e-04, 9.9950e-01],
             [9.9950e-01, 2.2158e-04, 2.7776e-04],
             [9.9957e-01, 2.1549e-04, 2.1694e-04],
             [9.9954e-01, 1.5746e-04, 3.0039e-04],
             [9.9966e-01, 1.5534e-04, 1.8776e-04],
             [1.9792e-02, 2.2271e-04, 9.7999e-01],
             [2.8085e-04, 7.3651e-05, 9.9965e-01],
             [1.4179e-04, 6.8673e-05, 9.9979e-01],
             [2.7716e-04, 7.0701e-05, 9.9965e-01],
             [4.3783e-03, 2.8395e-04, 9.9534e-01],
             [1.4646e-04, 7.1869e-05, 9.9978e-01],
             [6.6058e-01, 1.6691e-03, 3.3775e-01],
             [9.7598e-01, 1.0990e-03, 2.2922e-02],
             [6.7642e-01, 1.3222e-03, 3.2225e-01],
             [1.4284e-01, 1.1195e-03, 8.5604e-01],
             [1.0736e-02, 1.3971e-03, 9.8787e-01],
             [9.9259e-01, 8.8906e-04, 6.5207e-03],
             [9.9477e-01, 6.1016e-04, 4.6163e-03],
             [9.8733e-01, 1.2583e-03, 1.1409e-02],
             [2.7577e-04, 2.8709e-04, 9.9944e-01],
             [9.8108e-05, 1.3244e-04, 9.9977e-01],
             [7.7432e-05, 1.5000e-04, 9.9977e-01],
             [8.7345e-05, 1.0363e-04, 9.9981e-01],
             [9.1520e-05, 1.3899e-04, 9.9977e-01],
             [7.2311e-05, 1.0865e-04, 9.9982e-01],
             [1.0688e-04, 1.7087e-03, 9.9818e-01],
             [1.4058e-04, 9.9785e-01, 2.0130e-03],
             [1.3697e-04, 9.9747e-01, 2.3910e-03],
             [3.6230e-04, 9.7732e-01, 2.2315e-02],
             [6.3310e-05, 7.4447e-05, 9.9986e-01],
             [6.0260e-05, 9.9677e-05, 9.9984e-01],
             [6.2579e-05, 8.8337e-05, 9.9985e-01],
             [1.5951e-04, 9.9714e-01, 2.6995e-03],
             [3.7151e-04, 9.9490e-01, 4.7318e-03],
             [1.5582e-04, 9.9739e-01, 2.4536e-03],
             [3.0166e-04, 1.9242e-01, 8.0728e-01],
             [8.8418e-05, 8.4392e-04, 9.9907e-01],
             [2.3250e-04, 9.4894e-01, 5.0831e-02],
             [9.0127e-04, 9.5044e-01, 4.8656e-02],
             [1.1057e-04, 9.9943e-01, 4.5540e-04],
             [1.0952e-04, 9.9941e-01, 4.7737e-04],
             [1.0293e-04, 9.9965e-01, 2.4838e-04],
             [1.3366e-04, 9.9904e-01, 8.2613e-04],
             [2.6239e-04, 3.4179e-04, 9.9940e-01],
             [9.9899e-01, 4.5507e-04, 5.5769e-04],
             [9.9940e-01, 3.4406e-04, 2.6095e-04],
             [9.9912e-01, 2.6102e-04, 6.1583e-04],
             [9.9900e-01, 7.1279e-04, 2.8394e-04],
             [9.9921e-01, 5.9424e-04, 1.9114e-04],
             [9.9943e-01, 4.6595e-04, 1.0437e-04],
             [9.9950e-01, 2.5900e-04, 2.3724e-04],
             [9.9957e-01, 2.5886e-04, 1.7324e-04],
             [9.9553e-01, 4.6928e-04, 3.9974e-03],
             [6.7241e-05, 9.5410e-05, 9.9984e-01],
             [6.3152e-05, 1.0031e-04, 9.9984e-01],
             [5.9460e-05, 9.9488e-05, 9.9984e-01],
             [8.9533e-05, 1.8740e-04, 9.9972e-01],
             [3.5589e-04, 9.9382e-01, 5.8234e-03],
             [2.7958e-04, 9.7743e-01, 2.2292e-02],
             [1.3352e-04, 9.9752e-01, 2.3453e-03],
             [1.6707e-04, 9.9876e-01, 1.0775e-03],
             [1.2522e-04, 9.9908e-01, 7.9147e-04],
             [1.9803e-04, 9.9684e-01, 2.9615e-03],
             [9.8399e-05, 1.8405e-04, 9.9972e-01],
             [9.0291e-05, 9.9874e-01, 1.1677e-03],
             [7.3794e-05, 9.9920e-01, 7.2704e-04],
             [1.2687e-04, 9.9617e-01, 3.7001e-03],
             [2.6345e-04, 1.0121e-03, 9.9872e-01]]], device='cuda:0'), 'last_hidden_state': tensor([[[ 0.6225, -0.0142,  0.3417,  ..., -0.8350,  0.6950,  0.0122],
             [ 0.5098,  0.2632,  0.4508,  ..., -0.0542, -0.3744,  0.3085],
             [ 0.3304, -0.0706,  0.5190,  ..., -0.4323, -0.4408,  0.3608],
             ...,
             [ 0.8815, -0.2915,  0.4685,  ..., -0.6013,  0.7773, -0.2447],
             [ 0.8334, -0.3189,  0.4019,  ..., -0.6391,  0.8216, -0.2887],
             [ 0.7736, -0.3229,  0.3105,  ..., -0.7900,  0.8073, -0.3357]],
    
            [[ 0.8442,  0.0705,  0.5083,  ..., -0.4362,  0.8828, -0.2508],
             [ 0.8295, -0.0301,  0.4844,  ..., -0.8869,  0.0812,  0.1671],
             [ 0.9094,  0.3603,  0.7471,  ..., -0.5944,  0.2751, -0.3416],
             ...,
             [-0.7824,  0.1581, -0.8000,  ..., -0.2772,  0.5014,  0.1850],
             [-0.7882,  0.6704, -0.8703,  ..., -0.4553,  0.7861, -0.1652],
             [ 0.3163,  0.0580,  0.2595,  ..., -0.6415,  0.6288, -0.2301]]],
           device='cuda:0')}
    ['(((((.(((.(((((........)))).))).....((((........)))).)))))', '.......(((((......)))))...........((((......(((..(((.......)))...)))..)))))).(((((((((....)))))).)))']
    

Step 9. Pipeline Creation
~~~~~~~~~~~~~~~~~~~~~~~~~

The OmniGenome package provides pipelines for genomic FM development.
The pipeline can be used to train, fine-tune, and evaluate genomic FMs.
The pipeline can be used with a single command to train a genomic FM on
a dataset. The pipeline can also be used to fine-tune a pre-trained
genomic FM on a new dataset. The pipeline can be used to evaluate the
performance of a genomic FM on a dataset. The pipeline can be used to
generate predictions using a genomic FM.

.. code:: ipython3

    # from omnigenome import Pipeline, PipelineHub
    # 
    # pipeline = Pipeline(
    #     name="OmniGenome-186M-SSP-Pipeline",
    #     # model_name_or_path="OmniGenome-186M-SSP",  # The model name or path can be specified
    #     # tokenizer="OmniGenome-186M-SSP",  # The tokenizer can be specified
    #     model_name_or_path=ssp_model,
    #     tokenizer=ssp_model.tokenizer,
    #     datasets={
    #         "train": "toy_datasets/train.json",
    #         "test": "toy_datasets/test.json",
    #         "valid": "toy_datasets/valid.json",
    #     },
    #     trainer=trainer,
    #     device=ssp_model.model.device,
    # )

Using the Pipeline
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # results = pipeline(examples[0])
    # print(results)
    # 
    # pipeline.train()
    # 
    # pipeline.save("OmniGenome-186M-SSP-Pipeline", overwrite=True)
    # 
    # pipeline = PipelineHub.load("OmniGenome-186M-SSP-Pipeline")
    # results = pipeline(examples)
    # print(results)

Web Demo for RNA Secondary Structure Prediction
-----------------------------------------------

.. code:: ipython3

    import os
    import time
    import base64
    import tempfile
    from pathlib import Path
    import json
    import numpy as np
    import gradio as gr
    import RNA
    from omnigenome import ModelHub
    
    # 加载模型
    ssp_model = ModelHub.load("OmniGenome-186M-SSP")
    
    # 临时 SVG 存储目录
    TEMP_DIR = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {TEMP_DIR}")
    
    
    def ss_validity_loss(rna_strct: str) -> float:
        left = right = 0
        dots = rna_strct.count('.')
        for c in rna_strct:
            if c == '(':
                left += 1
            elif c == ')':
                if left:
                    left -= 1
                else:
                    right += 1
            elif c != '.':
                raise ValueError(f"Invalid char {c}")
        return (left + right) / (len(rna_strct) - dots + 1e-8)
    
    
    def find_invalid_positions(struct: str) -> list:
        stack, invalid = [], []
        for i, c in enumerate(struct):
            if c == '(': stack.append(i)
            elif c == ')':
                if stack:
                    stack.pop()
                else:
                    invalid.append(i)
        invalid.extend(stack)
        return invalid
    
    
    def generate_svg_datauri(rna_seq: str, struct: str) -> str:
        """生成 SVG 并返回 Base64 URI"""
        try:
            path = TEMP_DIR / f"{hash(rna_seq+struct)}.svg"
            RNA.svg_rna_plot(rna_seq, struct, str(path))
            time.sleep(0.1)
            svg_bytes = path.read_bytes()
            b64 = base64.b64encode(svg_bytes).decode('utf-8')
        except Exception as e:
            err = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
                   f'<text x="50" y="100" fill="red">Error: {e}</text></svg>')
            b64 = base64.b64encode(err.encode()).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64}"
    
    
    def fold(rna_seq: str, gt_struct: str):
        """展示 Ground Truth、ViennaRNA 与模型预测的结构对比"""
        if not rna_seq.strip():
            return "", "", "", ""
        # Ground Truth: 用户输入优先
        ground = gt_struct.strip() if gt_struct and gt_struct.strip() else ""
        gt_uri = generate_svg_datauri(rna_seq, ground) if ground else ""
    
        # ViennaRNA 预测
        vienna_struct, vienna_energy = RNA.fold(rna_seq)
        vienna_uri = generate_svg_datauri(rna_seq, vienna_struct)
    
        # 模型预测
        result = ssp_model.inference(rna_seq)
        pred = "".join(result.get('predictions', []))
        if ss_validity_loss(pred):
            for i in find_invalid_positions(pred):
                pred = pred[:i] + '.' + pred[i+1:]
        pred_uri = generate_svg_datauri(rna_seq, pred)
    
        # 统计信息
        match_gt = (sum(a==b for a,b in zip(ground, pred)) / len(ground)) if ground else 0
        match_vienna = sum(a==b for a,b in zip(vienna_struct, pred)) / len(vienna_struct)
        stats = (
            f"GT↔Pred Match: {match_gt:.2%}" + (" | " if ground else "") +
            f"Vienna↔Pred Match: {match_vienna:.2%}"
        )
    
        # 合并 HTML：三图水平排列
        combined = (
            '<div style="display:flex;justify-content:space-around;">'
            f'{f"<div><h4>Ground Truth</h4><img src=\"{gt_uri}\" style=\"max-width:100%;height:auto;\"/></div>" if ground else ""}'
            f'<div><h4>ViennaRNA</h4><img src=\"{vienna_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
            f'<div><h4>Prediction</h4><img src=\"{pred_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
            '</div>'
        )
        return ground, vienna_struct, pred, stats, combined
    
    
    def sample_rna_sequence():
        """从测试集中抽样，返回序列与 Ground Truth 结构"""
        try:
            exs = [json.loads(l) for l in open('toy_datasets/Archive2/test.json')]
            ex = exs[np.random.randint(len(exs))]
            return ex['seq'], ex.get('label','')
        except Exception as e:
            return f"加载样本出错: {e}", ""
    
    # Gradio UI
    with gr.Blocks(css="""
    .heading {text-align:center;color:#2a4365;}
    .controls {display:flex;gap:10px;margin:20px 0;}
    .status {padding:10px;background:#f0f4f8;border-radius:4px;white-space:pre;}
    """) as demo:
        gr.Markdown("# RNA 结构预测对比", elem_classes="heading")
        with gr.Row():
            rna_input = gr.Textbox(label="RNA 序列", lines=3)
            structure_input = gr.Textbox(label="Ground Truth 结构 (可选)", lines=3)
        with gr.Row(elem_classes="controls"):
            sample_btn = gr.Button("抽取样本")
            run_btn = gr.Button("预测并对比", variant="primary")
        stats_out    = gr.Textbox(label="统计信息", interactive=False, elem_classes="status")
        gt_out       = gr.Textbox(label="Ground Truth", interactive=False)
        vienna_out   = gr.Textbox(label="ViennaRNA 结构", interactive=False)
        pred_out     = gr.Textbox(label="Prediction 结构", interactive=False)
        combined_view= gr.HTML(label="三图对比视图")
    
        run_btn.click(
            fold,
            inputs=[rna_input, structure_input],
            outputs=[gt_out, vienna_out, pred_out, stats_out, combined_view]
        )
        sample_btn.click(
            sample_rna_sequence,
            outputs=[rna_input, structure_input]
        )
    
        demo.launch(share=True)
    


.. parsed-literal::

    ['.ipynb_checkpoints', 'annotated_structure.svg', 'auto_benchmark.py', 'benchmark', 'benchmarks_info.json', 'best_pred_struct.svg', 'easy_rna_design.py', 'eterna100_contrafold.txt', 'eterna100_vienna2.txt', 'eterna100_vienna2.txt.result', 'EternaV2_RNA_design_demo.py', 'mlm_augmentation.py', 'OmniGenome-186M-SSP', 'OmniGenome-186M-SSP-Pipeline', 'OmniGenome_RNA_design.ipynb', 'predicted_structure.svg', 'readme.md', 'real_structure.svg', 'rna_modeling_using_omnigenome.py', 'secondary_structure_prediction_demo.ipynb', 'ssp_inference.py', 'test.py', 'toy_datasets', 'true_struct.svg', 'zero_shot_secondary_structure_prediction.py']
    [2024-08-14 22:47:21] (0.0.8alpha) Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.0.8alpha', 'torch_version': '2.1.2+cu12.1+gita8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb', 'transformers_version': '4.42.0.dev0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'OmniSingleNucleotideTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: OmniGenome-186M-SSP
    Model Type: mprna
    Model Architecture: ['OmniGenomeModel']
    Model Parameters: 185.886801 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "OmniGenome-186M-SSP",
      "architectures": [
        "OmniGenomeModel"
      ],
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-186M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-186M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 720,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2560,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "metadata": {
        "library_name": "OmniGenome",
        "model_cls": "OmniGenomeModelForTokenClassification",
        "model_name": "OmniGenomeModelForTokenClassification",
        "omnigenome_version": "0.0.8alpha",
        "tokenizer_cls": "OmniSingleNucleotideTokenizer",
        "torch_version": "2.1.2+cu12.1+gita8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb",
        "transformers_version": "4.43.2"
      },
      "model_type": "mprna",
      "num_attention_heads": 30,
      "num_generation": 50,
      "num_hidden_layers": 32,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float16",
      "transformers_version": "4.42.0.dev0",
      "use_cache": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    

.. parsed-literal::

    C:\Users\chuan\miniconda3\lib\site-packages\gradio\routes.py:1019: DeprecationWarning: 
            on_event is deprecated, use lifespan event handlers instead.
    
            Read more about it in the
            [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
            
      @app.on_event("startup")
    C:\Users\chuan\miniconda3\lib\site-packages\fastapi\applications.py:4495: DeprecationWarning: 
            on_event is deprecated, use lifespan event handlers instead.
    
            Read more about it in the
            [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
            
      return self.router.on_event(event_type)
    

.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    IMPORTANT: You are using gradio version 4.25.0, however version 4.29.0 is available, please upgrade.
    --------
    Running on public URL: https://092094b2837cbc5f03.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
    

::


    ---------------------------------------------------------------------------

    ConnectTimeout                            Traceback (most recent call last)

    File ~\miniconda3\lib\site-packages\httpx\_transports\default.py:69, in map_httpcore_exceptions()
         68 try:
    ---> 69     yield
         70 except Exception as exc:
    

    File ~\miniconda3\lib\site-packages\httpx\_transports\default.py:233, in HTTPTransport.handle_request(self, request)
        232 with map_httpcore_exceptions():
    --> 233     resp = self._pool.handle_request(req)
        235 assert isinstance(resp.stream, typing.Iterable)
    

    File ~\miniconda3\lib\site-packages\httpcore\_sync\connection_pool.py:216, in ConnectionPool.handle_request(self, request)
        215     self._close_connections(closing)
    --> 216     raise exc from None
        218 # Return the response. Note that in this case we still have to manage
        219 # the point at which the response is closed.
    

    File ~\miniconda3\lib\site-packages\httpcore\_sync\connection_pool.py:196, in ConnectionPool.handle_request(self, request)
        194 try:
        195     # Send the request on the assigned connection.
    --> 196     response = connection.handle_request(
        197         pool_request.request
        198     )
        199 except ConnectionNotAvailable:
        200     # In some cases a connection may initially be available to
        201     # handle a request, but then become unavailable.
        202     #
        203     # In this case we clear the connection and try again.
    

    File ~\miniconda3\lib\site-packages\httpcore\_sync\connection.py:99, in HTTPConnection.handle_request(self, request)
         98     self._connect_failed = True
    ---> 99     raise exc
        101 return self._connection.handle_request(request)
    

    File ~\miniconda3\lib\site-packages\httpcore\_sync\connection.py:76, in HTTPConnection.handle_request(self, request)
         75 if self._connection is None:
    ---> 76     stream = self._connect(request)
         78     ssl_object = stream.get_extra_info("ssl_object")
    

    File ~\miniconda3\lib\site-packages\httpcore\_sync\connection.py:154, in HTTPConnection._connect(self, request)
        153 with Trace("start_tls", logger, request, kwargs) as trace:
    --> 154     stream = stream.start_tls(**kwargs)
        155     trace.return_value = stream
    

    File ~\miniconda3\lib\site-packages\httpcore\_backends\sync.py:168, in SyncStream.start_tls(self, ssl_context, server_hostname, timeout)
        167         self.close()
    --> 168         raise exc
        169 return SyncStream(sock)
    

    File ~\miniconda3\lib\contextlib.py:137, in _GeneratorContextManager.__exit__(self, typ, value, traceback)
        136 try:
    --> 137     self.gen.throw(typ, value, traceback)
        138 except StopIteration as exc:
        139     # Suppress StopIteration *unless* it's the same exception that
        140     # was passed to throw().  This prevents a StopIteration
        141     # raised inside the "with" statement from being suppressed.
    

    File ~\miniconda3\lib\site-packages\httpcore\_exceptions.py:14, in map_exceptions(map)
         13     if isinstance(exc, from_exc):
    ---> 14         raise to_exc(exc) from exc
         15 raise
    

    ConnectTimeout: _ssl.c:1112: The handshake operation timed out

    
    The above exception was the direct cause of the following exception:
    

    ConnectTimeout                            Traceback (most recent call last)

    Cell In[1], line 204
        195     repair_button.click(
        196         fn=repair_rna_structure,
        197         inputs=[rna_input, pred_structure_output],
        198         outputs=[pred_structure_output, predicted_image],
        199     )
        201     sample_button.click(
        202         fn=sample_rna_sequence, outputs=[rna_input, strcut_input, anno_structure_output]
        203     )
    --> 204 demo.launch(share=True)
    

    File ~\miniconda3\lib\site-packages\gradio\blocks.py:2283, in Blocks.launch(self, inline, inbrowser, share, debug, max_threads, auth, auth_message, prevent_thread_lock, show_error, server_name, server_port, height, width, favicon_path, ssl_keyfile, ssl_certfile, ssl_keyfile_password, ssl_verify, quiet, show_api, allowed_paths, blocked_paths, root_path, app_kwargs, state_session_capacity, share_server_address, share_server_protocol, auth_dependency, _frontend)
       2280 from IPython.display import HTML, Javascript, display  # type: ignore
       2282 if self.share and self.share_url:
    -> 2283     while not networking.url_ok(self.share_url):
       2284         time.sleep(0.25)
       2285     artifact = HTML(
       2286         f'<div><iframe src="{self.share_url}" width="{self.width}" height="{self.height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
       2287     )
    

    File ~\miniconda3\lib\site-packages\gradio\networking.py:53, in url_ok(url)
         51 with warnings.catch_warnings():
         52     warnings.filterwarnings("ignore")
    ---> 53     r = httpx.head(url, timeout=3, verify=False)
         54 if r.status_code in (200, 401, 302):  # 401 or 302 if auth is set
         55     return True
    

    File ~\miniconda3\lib\site-packages\httpx\_api.py:278, in head(url, params, headers, cookies, auth, proxy, proxies, follow_redirects, cert, verify, timeout, trust_env)
        255 def head(
        256     url: URLTypes,
        257     *,
       (...)
        268     trust_env: bool = True,
        269 ) -> Response:
        270     """
        271     Sends a `HEAD` request.
        272 
       (...)
        276     on this function, as `HEAD` requests should not include a request body.
        277     """
    --> 278     return request(
        279         "HEAD",
        280         url,
        281         params=params,
        282         headers=headers,
        283         cookies=cookies,
        284         auth=auth,
        285         proxy=proxy,
        286         proxies=proxies,
        287         follow_redirects=follow_redirects,
        288         cert=cert,
        289         verify=verify,
        290         timeout=timeout,
        291         trust_env=trust_env,
        292     )
    

    File ~\miniconda3\lib\site-packages\httpx\_api.py:106, in request(method, url, params, content, data, files, json, headers, cookies, auth, proxy, proxies, timeout, follow_redirects, verify, cert, trust_env)
         46 """
         47 Sends an HTTP request.
         48 
       (...)
         95 ```
         96 """
         97 with Client(
         98     cookies=cookies,
         99     proxy=proxy,
       (...)
        104     trust_env=trust_env,
        105 ) as client:
    --> 106     return client.request(
        107         method=method,
        108         url=url,
        109         content=content,
        110         data=data,
        111         files=files,
        112         json=json,
        113         params=params,
        114         headers=headers,
        115         auth=auth,
        116         follow_redirects=follow_redirects,
        117     )
    

    File ~\miniconda3\lib\site-packages\httpx\_client.py:827, in Client.request(self, method, url, content, data, files, json, params, headers, cookies, auth, follow_redirects, timeout, extensions)
        812     warnings.warn(message, DeprecationWarning)
        814 request = self.build_request(
        815     method=method,
        816     url=url,
       (...)
        825     extensions=extensions,
        826 )
    --> 827 return self.send(request, auth=auth, follow_redirects=follow_redirects)
    

    File ~\miniconda3\lib\site-packages\httpx\_client.py:914, in Client.send(self, request, stream, auth, follow_redirects)
        906 follow_redirects = (
        907     self.follow_redirects
        908     if isinstance(follow_redirects, UseClientDefault)
        909     else follow_redirects
        910 )
        912 auth = self._build_request_auth(request, auth)
    --> 914 response = self._send_handling_auth(
        915     request,
        916     auth=auth,
        917     follow_redirects=follow_redirects,
        918     history=[],
        919 )
        920 try:
        921     if not stream:
    

    File ~\miniconda3\lib\site-packages\httpx\_client.py:942, in Client._send_handling_auth(self, request, auth, follow_redirects, history)
        939 request = next(auth_flow)
        941 while True:
    --> 942     response = self._send_handling_redirects(
        943         request,
        944         follow_redirects=follow_redirects,
        945         history=history,
        946     )
        947     try:
        948         try:
    

    File ~\miniconda3\lib\site-packages\httpx\_client.py:979, in Client._send_handling_redirects(self, request, follow_redirects, history)
        976 for hook in self._event_hooks["request"]:
        977     hook(request)
    --> 979 response = self._send_single_request(request)
        980 try:
        981     for hook in self._event_hooks["response"]:
    

    File ~\miniconda3\lib\site-packages\httpx\_client.py:1015, in Client._send_single_request(self, request)
       1010     raise RuntimeError(
       1011         "Attempted to send an async request with a sync Client instance."
       1012     )
       1014 with request_context(request=request):
    -> 1015     response = transport.handle_request(request)
       1017 assert isinstance(response.stream, SyncByteStream)
       1019 response.request = request
    

    File ~\miniconda3\lib\site-packages\httpx\_transports\default.py:233, in HTTPTransport.handle_request(self, request)
        220 req = httpcore.Request(
        221     method=request.method,
        222     url=httpcore.URL(
       (...)
        230     extensions=request.extensions,
        231 )
        232 with map_httpcore_exceptions():
    --> 233     resp = self._pool.handle_request(req)
        235 assert isinstance(resp.stream, typing.Iterable)
        237 return Response(
        238     status_code=resp.status,
        239     headers=resp.headers,
        240     stream=ResponseStream(resp.stream),
        241     extensions=resp.extensions,
        242 )
    

    File ~\miniconda3\lib\contextlib.py:137, in _GeneratorContextManager.__exit__(self, typ, value, traceback)
        135     value = typ()
        136 try:
    --> 137     self.gen.throw(typ, value, traceback)
        138 except StopIteration as exc:
        139     # Suppress StopIteration *unless* it's the same exception that
        140     # was passed to throw().  This prevents a StopIteration
        141     # raised inside the "with" statement from being suppressed.
        142     return exc is not value
    

    File ~\miniconda3\lib\site-packages\httpx\_transports\default.py:86, in map_httpcore_exceptions()
         83     raise
         85 message = str(exc)
    ---> 86 raise mapped_exc(message) from exc
    

    ConnectTimeout: _ssl.c:1112: The handshake operation timed out


Conclusion
~~~~~~~~~~

In this demonstration, we have shown how to fine-tune a genomic
foundation model for RNA secondary structure prediction using the
OmniGenome package. We have also shown how to use the trained model for
inference and how to create a web demo for RNA secondary structure
prediction. We hope this demonstration will help you get started with
genomic foundation model development using OmniGenome.

