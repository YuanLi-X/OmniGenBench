Auto-Benchmarking
==========================================

This script is used to auto-benchmark the Genomic Foundation Model on
diversified downstream tasks. We have automated the benchmark pipeline
based on the OmniGenome package. Once your foundation model is trained,
you can use this script to evaluate the performance of the model. The
script will automatically load the datasets, preprocess the data, and
evaluate the model on the tasks. The script will output the performance
of the model on each task.

[Optional] Prepare your own benchmark datasets
----------------------------------------------

We have provided a set of benchmark datasets in the tutorials, you can
use them to evaluate the performance of the model. If you want to
evaluate the model on your own datasets, you can prepare the datasets in
the following steps: 1. Prepare the datasets in the following format: -
The datasets should be in the ``json`` format. - The datasets should
contain two columns: ``sequence`` and ``label``. - The ``sequence``
column should contain the DNA sequences. - The ``label`` column should
contain the labels of the sequences. 2. Save the datasets in a folder
like the existing benchmark datasets. This folder is referred to as the
``root`` in the script. 3. Place the model and tokenizer in an
accessible folder. 4. Sometimes the tokenizer does not work well with
the datasets, you can write a custom tokenizer and model wrapper in the
``omnigenome_wrapper.py`` file. More detailed documentation on how to
write the custom tokenizer and model wrapper will be provided.

Prepare the benchmark environment
---------------------------------

Before running the benchmark, you need to install the following required
packages in addition to PyTorch and other dependencies. Find the
installation instructions for PyTorch at
https://pytorch.org/get-started/locally/.

.. code:: bash

   pip install omnigenome, findfile, autocuda, metric-visualizer, transformers

Import the required packages
----------------------------

.. code:: ipython3

    from omnigenome import AutoBench
    import autocuda


.. parsed-literal::

    C:\Users\chuan\miniconda3\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

.. parsed-literal::

                           
        **@@ #========= @@**            ___                     _ 
          **@@ +----- @@**             / _ \  _ __ ___   _ __  (_)
            **@@ = @@**               | | | || '_ ` _ \ | '_ \ | |
               **@@                   | |_| || | | | | || | | || |
            @@** = **@@                \___/ |_| |_| |_||_| |_||_|
         @@** ------+ **@@                
       @@** =========# **@@            ____  
      @@ ---------------+ @@          / ___|  ___  _ __    ___   _ __ ___    ___ 
     @@ ================== @@        | |  _  / _ \| '_ \  / _ \ | '_ ` _ \  / _ \
      @@ +--------------- @@         | |_| ||  __/| | | || (_) || | | | | ||  __/ 
       @@** #========= **@@           \____| \___||_| |_| \___/ |_| |_| |_| \___| 
        @@** +------ **@@          
           @@** = **@@           
              @@**                    ____                      _   
           **@@ = @@**               | __ )   ___  _ __    ___ | |__  
        **@@ -----+  @@**            |  _ \  / _ \| '_ \  / __|| '_ \ 
      **@@ ==========# @@**          | |_) ||  __/| | | || (__ | | | |
      @@ --------------+ @@**        |____/  \___||_| |_| \___||_| |_|
    
    

1. Define the root folder of the benchmark datasets
---------------------------------------------------

Define the root where the benchmark datasets are stored.

.. code:: ipython3

    root = 'RGB'  # Abbreviation of the RNA genome benchmark

2. Define the model and tokenizer paths
---------------------------------------

Provide the path to the model and tokenizer.

.. code:: ipython3

    model_name_or_path = 'anonymous8/OmniGenome-52M'



3. Initialize the AutoBench
---------------------------

Select the available CUDA device based on your hardware.

.. code:: ipython3

    device = autocuda.auto_cuda()
    auto_bench = AutoBench(
        benchmark=root,
        model_name_or_path=model_name_or_path,
        device="cuda",
        overwrite=True,
    )


.. parsed-literal::

    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Benchmark: RGB does not exist. Search online for available benchmarks.
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Loaded benchmarks:  ['RNA-mRNA', 'RNA-SNMD', 'RNA-SNMR', 'RNA-SSP-Archive2', 'RNA-SSP-rnastralign', 'RNA-SSP-bpRNA', 'RNA-TE-Prediction.Arabidopsis', 'RNA-TE-Prediction.Rice', 'RNA-Region-Classification.Arabidopsis', 'RNA-Region-Classification.Rice']
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Benchmark Root: __OMNIGENOME_DATA__/benchmarks/RGB
    Benchmark List: ['RNA-mRNA', 'RNA-SNMD', 'RNA-SNMR', 'RNA-SSP-Archive2', 'RNA-SSP-rnastralign', 'RNA-SSP-bpRNA', 'RNA-TE-Prediction.Arabidopsis', 'RNA-TE-Prediction.Rice', 'RNA-Region-Classification.Arabidopsis', 'RNA-Region-Classification.Rice']
    Model Name or Path: OmniGenome-52M
    Tokenizer: None
    Metric Visualizer Path: ./autobench_evaluations/RGB-OmniGenome-52M-20250419_171940.mv
    BenchConfig Details: <module 'bench_metadata' from 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB/metadata.py'>
    
    

4. Run the benchmark
--------------------

The downstream tasks have predefined configurations for fair comparison.
However, sometimes you might need to adjust the configuration based on
your dataset or resources. For instance, adjusting the ``max_length`` or
batch size. To adjust the configuration, you can override parameters in
the ``AutoBenchConfig`` class.

.. code:: ipython3

    batch_size = 4
    epochs = 1  # increase for real cases
    seeds = [42]
    auto_bench.run(epochs=epochs, batch_size=batch_size, seeds=seeds)


.. parsed-literal::

    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-mRNA Progress:  1 / 10 10.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-mRNA\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-mRNA\\__pycache__\\config.cpython-312.pyc', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-mRNA\\__pycache__\\config.cpython-39.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA\config.py>
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-mRNA from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA\config.py
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-mRNA', 'task_type': 'token_regression', 'label2id': None, 'num_labels': 3, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 110, 'seeds': [45, 46, 47], 'compute_metrics': [<function RegressionMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA8A1080>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-mRNA/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-mRNA/test.json', 'valid_file': None, 'dataset_cls': <class 'config.Dataset'>, 'model_cls': <class 'omnigenome.src.model.regression.model.OmniGenomeModelForTokenRegression'>}
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:19:40] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-mRNA: task_name: RNA-mRNA
    task_type: token_regression
    label2id: None
    num_labels: 3
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 110
    seeds: [42]
    compute_metrics: [<function RegressionMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA8A1080>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/test.json
    valid_file: None
    dataset_cls: <class 'config.Dataset'>
    model_cls: <class 'omnigenome.src.model.regression.model.OmniGenomeModelForTokenRegression'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:19:45] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenRegression
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenRegression', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenRegression'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": null,
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:19:45] [OmniGenome 0.2.4alpha4]  Detected max_length=110 in the dataset, using it as the max_length.
    [2025-04-19 17:19:45] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/train.json...
    [2025-04-19 17:19:45] [OmniGenome 0.2.4alpha4]  Loaded 1728 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/train.json
    [2025-04-19 17:19:45] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 1728/1728 [00:01<00:00, 1016.65it/s]
    

.. parsed-literal::

    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=110, label_padding_length=110
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 109.0, 'max_seq_len': 109, 'min_seq_len': 109, 'avg_label_len': 110.0, 'max_label_len': 110, 'min_label_len': 110}
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 4, 4, 4, 9, 4, 4, 9, 9, 4, 5, 5, 6, 9, 6, 5, 5, 9, 5, 5, 4,
            5, 6, 4, 4, 4, 6, 9, 4, 6, 6, 6, 4, 5, 6, 5, 5, 4, 4, 9, 5, 9, 5, 5, 4,
            9, 6, 6, 5, 6, 6, 4, 4, 6, 5, 5, 9, 6, 4, 5, 6, 6, 9, 9, 4, 4, 6, 5, 4,
            9, 6, 4, 6, 9, 9, 5, 6, 5, 9, 5, 4, 9, 6, 5, 4, 4, 4, 4, 6, 4, 4, 4, 5,
            4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 2, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 'labels': tensor([[ 3.2630e-01,  2.3970e-01,  3.2620e-01],
            [ 1.1321e+00,  2.2392e+00,  1.8816e+00],
            [ 1.5661e+00,  1.0323e+00,  2.0296e+00],
            [ 1.4618e+00,  1.0172e+00,  1.3011e+00],
            [ 1.0462e+00,  1.1377e+00,  1.0372e+00],
            [ 2.1425e+00,  1.6494e+00,  2.0762e+00],
            [ 1.6369e+00,  3.7634e+00,  2.8956e+00],
            [ 1.3015e+00,  9.6770e-01,  1.2184e+00],
            [ 1.0998e+00,  2.6416e+00,  2.9895e+00],
            [ 7.9980e-01,  5.1540e-01,  3.1180e-01],
            [ 9.9050e-01,  9.3530e-01,  6.1890e-01],
            [ 3.3470e-01,  5.8290e-01,  4.6540e-01],
            [ 5.8100e-02,  1.5860e-01,  1.3990e-01],
            [ 4.0600e-02,  1.3210e-01,  1.5990e-01],
            [ 8.1500e-02,  2.6800e-01,  1.6590e-01],
            [ 1.1440e-01,  1.1540e-01,  1.2260e-01],
            [ 4.5720e-01,  2.5660e-01,  2.2420e-01],
            [ 1.4700e-02,  1.0760e-01,  1.2650e-01],
            [ 4.3730e-01,  3.3510e-01,  4.4240e-01],
            [ 2.5370e-01,  2.7200e-02,  1.3030e-01],
            [ 3.7300e-02,  9.3700e-02,  2.5280e-01],
            [ 2.8070e-01,  2.2370e-01,  3.6240e-01],
            [ 1.5120e-01,  2.1150e-01,  1.6910e-01],
            [ 1.0490e-01,  2.4100e-01,  2.8840e-01],
            [ 8.2500e-01,  8.1870e-01,  6.6950e-01],
            [ 1.0446e+00,  7.2280e-01,  8.0820e-01],
            [ 4.9550e-01,  4.1710e-01,  5.5470e-01],
            [ 8.9480e-01,  6.0210e-01,  3.9930e-01],
            [ 6.4510e-01,  1.5076e+00,  9.7660e-01],
            [ 5.7040e-01,  2.5980e-01,  4.5550e-01],
            [ 6.2200e-02,  4.0400e-02,  6.9300e-02],
            [ 3.6100e-02,  2.3090e-01,  1.3050e-01],
            [ 7.5700e-02,  1.2660e-01,  1.3670e-01],
            [ 2.8800e-01,  4.0330e-01,  2.9320e-01],
            [ 1.3020e-01,  3.0660e-01,  1.1650e-01],
            [ 3.8100e-02,  9.9500e-02,  8.9500e-02],
            [ 8.1900e-02,  1.8830e-01,  1.0500e-02],
            [-1.4800e-02,  1.4590e-01,  7.3000e-03],
            [ 8.7000e-03,  2.6960e-01,  2.1270e-01],
            [ 3.5250e-01,  4.3150e-01,  3.5650e-01],
            [ 1.7514e+00,  1.2023e+00,  1.1414e+00],
            [ 2.1660e-01,  2.0070e-01,  1.9790e-01],
            [ 5.0930e-01,  6.8050e-01,  7.3120e-01],
            [ 4.4180e-01,  4.8430e-01,  5.5360e-01],
            [ 9.3400e-02,  4.3880e-01,  6.3690e-01],
            [ 2.6600e-01,  4.8380e-01,  3.6260e-01],
            [ 3.5860e-01,  7.5980e-01,  6.9870e-01],
            [ 5.1200e-02,  1.3100e-01,  6.4700e-02],
            [ 2.7000e-02,  2.2410e-01,  6.6800e-02],
            [ 6.9000e-03,  6.5700e-02,  4.8500e-02],
            [-1.6600e-02,  5.3200e-02,  1.2500e-02],
            [ 1.2400e-02,  9.2200e-02,  6.8500e-02],
            [ 5.2320e-01,  7.7390e-01,  9.6540e-01],
            [ 6.1710e-01,  6.4470e-01,  1.0076e+00],
            [ 4.9080e-01,  2.6400e-01,  1.9450e-01],
            [ 5.3300e-01,  2.6050e-01,  2.8890e-01],
            [ 5.0100e-02,  9.0000e-02,  1.1630e-01],
            [ 5.2200e-02,  2.0330e-01,  3.4810e-01],
            [ 1.5450e-01,  6.3200e-02,  2.0800e-02],
            [ 4.4850e-01,  3.5290e-01,  3.1120e-01],
            [ 1.1230e-01,  1.4330e-01,  1.0750e-01],
            [ 3.1600e-02,  1.2630e-01,  9.3700e-02],
            [ 8.9300e-02,  3.8730e-01,  1.9210e-01],
            [ 2.5920e-01,  1.8314e+00,  1.3322e+00],
            [ 2.9410e-01,  1.4359e+00,  1.0734e+00],
            [ 5.9430e-01,  5.8220e-01,  7.6650e-01],
            [ 4.2140e-01,  7.1010e-01,  5.5010e-01],
            [ 3.7030e-01,  7.4120e-01,  3.7880e-01],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02]])}
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 4, 4, 5, 6, 6, 5, 9, 4, 6, 6, 5, 6, 4, 6, 6, 5, 4, 9, 4, 6,
            6, 5, 6, 6, 5, 4, 6, 9, 6, 6, 6, 4, 6, 6, 9, 6, 5, 6, 9, 6, 5, 4, 9, 5,
            6, 5, 4, 5, 9, 5, 4, 5, 4, 4, 5, 9, 5, 6, 5, 9, 6, 5, 4, 9, 4, 5, 4, 6,
            5, 5, 9, 9, 9, 9, 5, 6, 4, 4, 6, 6, 5, 9, 6, 4, 4, 4, 4, 6, 4, 4, 4, 5,
            4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 2, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 'labels': tensor([[ 5.1350e-01,  2.3210e-01,  5.0290e-01],
            [ 1.5078e+00,  2.4418e+00,  2.3320e+00],
            [ 8.8270e-01,  8.4520e-01,  1.2860e+00],
            [ 1.0481e+00,  5.2400e-01,  1.2027e+00],
            [ 2.2870e-01,  7.8450e-01,  1.1479e+00],
            [ 1.2460e-01,  1.2230e-01,  4.2740e-01],
            [ 1.9090e-01,  2.4880e-01,  3.4830e-01],
            [ 2.3430e-01,  4.0690e-01,  1.8000e-01],
            [ 1.3300e-01,  9.6520e-01,  4.9360e-01],
            [ 3.9150e-01,  5.8300e-02,  3.1320e-01],
            [ 2.9190e-01,  1.2940e-01,  3.0550e-01],
            [ 6.0070e-01,  7.3840e-01,  5.5870e-01],
            [ 3.0000e-01,  5.0050e-01,  4.1620e-01],
            [ 9.4300e-02,  4.0080e-01,  2.1310e-01],
            [ 6.7670e-01,  5.4230e-01,  5.7080e-01],
            [ 1.3160e-01,  1.4640e-01,  3.5930e-01],
            [ 5.8910e-01,  5.2890e-01,  3.2760e-01],
            [ 3.3770e-01,  7.3990e-01,  6.0060e-01],
            [ 3.2870e-01,  5.9820e-01,  5.9880e-01],
            [ 1.3200e+00,  2.2901e+00,  2.3748e+00],
            [ 6.7820e-01,  4.4340e-01,  3.4720e-01],
            [ 2.0750e-01,  2.9780e-01,  5.8980e-01],
            [ 3.2950e-01,  5.3550e-01,  4.8200e-01],
            [ 2.0700e-01,  4.5420e-01,  2.3280e-01],
            [ 5.8700e-02,  4.1380e-01,  1.8920e-01],
            [ 4.3810e-01,  5.2540e-01,  3.8950e-01],
            [ 3.4100e-02,  7.9900e-02,  1.0410e-01],
            [ 4.6000e-02,  2.3210e-01,  7.0700e-02],
            [ 1.3580e-01,  2.1150e-01,  8.9200e-02],
            [ 4.5500e-02,  2.2510e-01,  9.3400e-02],
            [ 0.0000e+00,  8.3900e-02,  0.0000e+00],
            [ 4.5400e-02,  5.5700e-02,  0.0000e+00],
            [ 1.8000e-01,  2.4660e-01,  6.9800e-02],
            [ 5.2630e-01,  3.2160e-01,  1.1580e-01],
            [ 1.3070e-01,  5.3400e-02,  9.2300e-02],
            [ 2.1560e-01,  7.9700e-02,  6.9000e-02],
            [ 1.8040e-01,  2.0100e-01,  5.4900e-02],
            [ 6.3900e-02,  5.2200e-02,  4.5800e-02],
            [ 7.2900e-02,  4.3500e-02,  1.0000e-01],
            [ 2.1200e-02,  2.0510e-01,  9.0900e-02],
            [ 1.7700e-01,  1.4370e-01,  1.4400e-01],
            [ 5.5170e-01,  2.2490e-01,  2.0140e-01],
            [ 5.4040e-01,  2.9240e-01,  1.3590e-01],
            [ 4.1140e-01,  4.4790e-01,  3.5160e-01],
            [ 1.5559e+00,  9.4520e-01,  7.5470e-01],
            [ 3.1180e-01,  4.8520e-01,  1.4020e-01],
            [ 3.9680e-01,  1.2616e+00,  6.7190e-01],
            [ 5.1700e-02,  1.7340e-01,  2.0400e-02],
            [ 0.0000e+00,  1.5260e-01,  0.0000e+00],
            [ 0.0000e+00,  1.1350e-01,  4.0700e-02],
            [ 2.8000e-02,  3.1610e-01,  9.3700e-02],
            [ 1.6390e-01,  5.6800e-02,  1.2700e-02],
            [ 2.1730e-01,  2.4580e-01,  7.9200e-02],
            [ 3.5220e-01,  1.8770e-01,  2.2500e-01],
            [ 1.6590e-01,  3.3160e-01,  2.9790e-01],
            [ 5.8050e-01,  2.3950e-01,  1.6700e-01],
            [ 1.6560e-01,  1.3540e-01,  1.1400e-02],
            [ 6.6700e-02,  1.7530e-01,  1.0220e-01],
            [ 1.5700e-02,  4.9900e-02,  1.9300e-02],
            [ 1.5560e-01,  2.1320e-01,  1.5350e-01],
            [ 1.5500e-02,  1.4610e-01,  5.7400e-02],
            [-7.8000e-03,  1.1210e-01,  1.0800e-02],
            [-1.1700e-02,  1.3510e-01,  6.2500e-02],
            [ 6.2000e-02,  1.4990e-01,  2.2650e-01],
            [ 5.4100e-02,  4.3630e-01,  2.0590e-01],
            [ 9.6020e-01,  1.3602e+00,  1.1152e+00],
            [ 1.2813e+00,  2.3379e+00,  1.3669e+00],
            [ 9.4800e-02,  2.0907e+00,  1.4542e+00],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02]])}
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  Detected max_length=110 in the dataset, using it as the max_length.
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/test.json...
    [2025-04-19 17:19:47] [OmniGenome 0.2.4alpha4]  Loaded 192 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-mRNA/test.json
    

.. parsed-literal::

    100%|██████████| 192/192 [00:00<00:00, 1045.15it/s]
    

.. parsed-literal::

    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=110, label_padding_length=110
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 109.0, 'max_seq_len': 109, 'min_seq_len': 109, 'avg_label_len': 110.0, 'max_label_len': 110, 'min_label_len': 110}
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 4, 4, 6, 9, 9, 6, 6, 4, 5, 9, 6, 9, 9, 9, 9, 6, 4, 9, 9, 6,
            6, 9, 4, 6, 4, 9, 9, 9, 6, 4, 6, 5, 4, 4, 4, 6, 5, 9, 9, 4, 6, 4, 9, 9,
            9, 6, 9, 5, 4, 6, 9, 9, 4, 6, 6, 4, 9, 6, 6, 9, 5, 9, 6, 4, 5, 5, 4, 6,
            6, 9, 9, 9, 9, 9, 5, 6, 4, 4, 6, 5, 9, 9, 6, 4, 4, 4, 4, 6, 4, 4, 4, 5,
            4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 2, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 'labels': tensor([[ 8.2280e-01,  1.7252e+00,  9.3210e-01],
            [ 2.0374e+00,  2.9090e+00,  3.0696e+00],
            [ 1.6742e+00,  1.1519e+00,  1.7692e+00],
            [ 8.7970e-01,  5.1380e-01,  1.0703e+00],
            [ 8.8900e-01,  1.3861e+00,  9.3910e-01],
            [ 1.8250e-01,  1.0246e+00,  8.3200e-01],
            [ 2.8570e-01,  1.0035e+00,  1.0496e+00],
            [ 6.5300e-02,  2.5100e-01,  1.5820e-01],
            [ 4.8800e-02,  2.7370e-01,  7.0100e-02],
            [ 6.3500e-02,  6.5500e-01,  1.8620e-01],
            [ 6.4600e-02,  4.5550e-01,  8.6500e-02],
            [-8.7000e-03,  5.1000e-02,  6.6500e-02],
            [ 2.7700e-02,  2.3880e-01,  7.9700e-02],
            [ 2.8000e-02,  6.4820e-01,  4.3350e-01],
            [ 3.2100e-02,  2.2160e-01,  1.9240e-01],
            [ 1.5700e-02,  1.8960e-01,  4.0200e-02],
            [ 1.3450e-01,  1.0220e-01,  3.7450e-01],
            [ 3.1800e-02,  4.5500e-02,  1.4690e-01],
            [ 1.5900e-02,  3.7710e-01,  9.7500e-02],
            [ 6.3400e-02,  1.5370e-01,  1.1330e-01],
            [ 6.3200e-02,  1.7360e-01,  1.1270e-01],
            [-8.4000e-03,  1.5700e-02,  6.3000e-03],
            [ 9.3200e-02,  2.4350e-01,  4.4600e-02],
            [ 6.1500e-02,  3.8170e-01,  1.5570e-01],
            [ 4.6800e-02,  1.4360e-01,  3.1800e-02],
            [-8.4000e-03,  5.5300e-02,  6.9500e-02],
            [ 4.6700e-02,  3.3930e-01,  1.2630e-01],
            [ 9.3000e-02,  2.3570e-01,  4.3400e-01],
            [ 2.1300e-02,  1.5680e-01,  1.8570e-01],
            [ 5.3300e-02,  5.2000e-02,  1.4290e-01],
            [ 4.6200e-02,  9.6000e-02,  0.0000e+00],
            [ 6.8300e-02,  2.0230e-01,  9.6600e-02],
            [ 7.0000e-03,  8.8100e-02,  6.6000e-02],
            [ 3.6100e-02,  2.5910e-01,  7.6200e-02],
            [ 3.9610e-01,  4.0680e-01,  1.3270e-01],
            [ 2.0621e+00,  9.5430e-01,  9.5620e-01],
            [ 2.0057e+00,  8.4390e-01,  4.6810e-01],
            [ 6.2070e-01,  1.5312e+00,  6.1390e-01],
            [ 1.4960e-01,  3.1250e-01,  1.6080e-01],
            [ 3.1600e-02,  2.0830e-01,  1.9700e-01],
            [ 2.9700e-02,  8.2400e-02,  7.4300e-02],
            [ 9.2000e-02,  1.7270e-01,  9.3200e-02],
            [ 2.3000e-02,  7.2700e-02,  5.3200e-02],
            [-1.0000e-02,  2.2990e-01,  6.7300e-02],
            [ 2.2900e-02,  1.9240e-01,  1.0540e-01],
            [-3.5000e-03,  2.6640e-01,  1.8350e-01],
            [ 3.2000e-03,  1.9240e-01,  1.2660e-01],
            [ 1.1400e-02,  1.1070e-01,  7.7600e-02],
            [ 1.1400e-02,  2.5380e-01,  7.7300e-02],
            [ 3.2000e-03,  7.2300e-02,  2.9000e-02],
            [ 1.3800e-02,  1.1440e-01,  7.3300e-02],
            [ 3.9600e-02,  1.1180e-01,  5.0900e-02],
            [ 4.8900e-02,  5.5930e-01,  1.6070e-01],
            [ 7.0500e-02,  9.5300e-02,  4.0900e-02],
            [ 6.5100e-02,  1.8200e-01,  4.6800e-02],
            [ 3.6500e-02,  1.3680e-01,  1.5600e-02],
            [ 3.3400e-02,  1.7350e-01,  1.2480e-01],
            [ 1.7100e-02,  3.0670e-01,  1.4220e-01],
            [ 2.0900e-02,  3.0150e-01,  2.9990e-01],
            [ 2.2100e-02,  5.1600e-02,  0.0000e+00],
            [-6.1500e-02,  8.9100e-02, -6.8000e-03],
            [-2.9270e-01,  3.3580e-01, -2.1740e-01],
            [-2.5030e-01,  5.0300e-02, -2.7790e-01],
            [-7.0000e-04,  1.8250e-01, -2.8100e-02],
            [-1.1920e-01, -9.9000e-02, -2.5420e-01],
            [ 3.1590e-01,  1.4300e-01, -1.4650e-01],
            [ 2.4000e-02,  2.9000e-01,  6.2000e-02],
            [ 4.6800e-02,  5.7780e-01,  1.0620e-01],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02]])}
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 4, 4, 9, 6, 5, 9, 4, 9, 5, 4, 6, 9, 9, 9, 9, 9, 9, 6, 6, 9,
            4, 5, 9, 5, 9, 6, 6, 4, 9, 4, 5, 4, 6, 4, 6, 4, 5, 9, 5, 9, 6, 6, 4, 9,
            4, 5, 4, 6, 4, 6, 6, 6, 4, 6, 4, 6, 5, 9, 6, 6, 9, 4, 6, 5, 4, 6, 6, 4,
            4, 6, 4, 5, 9, 9, 5, 6, 6, 9, 5, 9, 9, 5, 5, 4, 4, 4, 4, 6, 4, 4, 4, 5,
            4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 2, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 'labels': tensor([[ 1.4213e+00,  8.2220e-01,  4.2400e-01],
            [ 2.2508e+00,  3.2204e+00,  3.1675e+00],
            [ 1.8010e+00,  2.5420e-01,  5.1860e-01],
            [ 1.2016e+00,  4.5760e-01,  9.3910e-01],
            [ 1.0586e+00,  8.1530e-01,  9.0010e-01],
            [ 3.2190e-01,  3.6480e-01,  4.0820e-01],
            [ 7.4800e-02,  2.2220e-01,  1.2730e-01],
            [ 3.7300e-02,  1.4670e-01,  1.2670e-01],
            [-4.7100e-02,  1.1160e-01, -1.3100e-02],
            [ 0.0000e+00,  0.0000e+00,  4.2100e-02],
            [ 0.0000e+00,  7.2300e-02,  1.2560e-01],
            [ 0.0000e+00,  7.1900e-02,  0.0000e+00],
            [ 0.0000e+00,  7.1600e-02,  8.3400e-02],
            [ 1.1680e-01,  4.9550e-01,  4.1350e-01],
            [ 2.8110e-01,  1.0240e+00,  6.2150e-01],
            [ 4.2010e-01,  8.4690e-01,  1.1186e+00],
            [ 2.4200e-02,  1.5748e+00,  4.9950e-01],
            [ 4.8740e-01,  7.4030e-01,  5.0540e-01],
            [ 1.4299e+00,  1.2648e+00,  8.4970e-01],
            [ 3.6260e-01,  3.2870e-01,  1.5250e-01],
            [ 1.2697e+00,  1.2800e+00,  6.3790e-01],
            [ 3.4536e+00,  2.3998e+00,  1.3543e+00],
            [ 1.8039e+00,  9.5640e-01,  6.4750e-01],
            [ 4.4440e-01,  4.9340e-01,  2.7140e-01],
            [ 5.4100e-02,  1.2230e-01,  6.0200e-02],
            [ 0.0000e+00,  1.2130e-01,  0.0000e+00],
            [ 2.1600e-02,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  3.0000e-02],
            [ 4.3100e-02,  3.0300e-02,  8.9800e-02],
            [ 2.6120e-01,  3.7240e-01,  1.0120e-01],
            [ 7.9730e-01,  5.1220e-01,  4.3860e-01],
            [ 5.1820e-01,  4.6840e-01,  2.0300e-01],
            [ 3.6850e-01,  5.3880e-01,  2.1510e-01],
            [ 3.6400e-02,  1.3960e-01,  7.1100e-02],
            [ 1.9400e-02,  0.0000e+00,  2.8500e-02],
            [ 4.2000e-03,  1.9530e-01,  1.0600e-02],
            [ 4.2140e-01,  1.1690e-01,  1.0600e-02],
            [ 1.6787e+00,  5.8170e-01,  1.9190e-01],
            [ 8.3430e-01,  5.8730e-01,  2.8710e-01],
            [ 8.1820e-01,  2.8275e+00,  9.0140e-01],
            [ 2.0450e-01,  2.0400e-01,  1.5790e-01],
            [ 0.0000e+00,  1.6500e-01,  5.2500e-02],
            [ 0.0000e+00,  3.6600e-02,  2.6200e-02],
            [ 1.5700e-02,  9.0900e-02,  1.0450e-01],
            [ 9.3900e-02,  2.8520e-01,  1.0410e-01],
            [ 5.1860e-01,  3.9890e-01,  1.2940e-01],
            [ 7.4890e-01,  1.7760e-01,  2.6370e-01],
            [ 2.6110e-01,  5.1230e-01,  2.5310e-01],
            [ 7.1700e-02,  2.9680e-01,  8.3200e-02],
            [-1.5000e-02,  8.5800e-02,  5.7800e-02],
            [ 8.5700e-02,  2.2660e-01,  1.8180e-01],
            [ 0.0000e+00,  3.1600e-02,  4.9700e-02],
            [ 1.4400e-02,  4.7200e-02,  7.4400e-02],
            [ 2.8100e-02,  9.8700e-02,  3.2460e-01],
            [ 3.6500e-01,  2.8020e-01,  2.0050e-01],
            [ 5.6100e-02,  1.2150e-01,  3.5800e-01],
            [ 1.2200e-02,  1.1410e-01,  1.7790e-01],
            [ 0.0000e+00,  1.3410e-01,  9.4200e-02],
            [ 1.4000e-02,  1.4760e-01,  1.6380e-01],
            [ 2.7900e-02,  1.1710e-01,  7.0000e-02],
            [ 4.1800e-02,  4.3800e-02,  0.0000e+00],
            [-1.4900e-02,  1.0520e-01,  2.9100e-02],
            [ 4.1800e-02,  3.8090e-01,  1.6190e-01],
            [ 1.3900e-02,  1.2590e-01,  1.3800e-01],
            [-1.0000e-03,  1.0040e-01,  1.1980e-01],
            [ 2.7800e-02,  1.9180e-01,  1.5910e-01],
            [ 5.5400e-02,  2.3590e-01,  2.2510e-01],
            [ 9.5300e-02,  5.2840e-01,  3.1560e-01],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02],
            [-1.0000e+02, -1.0000e+02, -1.0000e+02]])}
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  Detected max_length=110 in the dataset, using it as the max_length.
    [2025-04-19 17:19:48] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Testing: 100%|██████████| 48/48 [00:03<00:00, 12.77it/s]
    

.. parsed-literal::

    [2025-04-19 17:19:52] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.99506414}
    [2025-04-19 17:19:52] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.99506414}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.4697: 100%|██████████| 432/432 [01:17<00:00,  5.56it/s]
    Testing: 100%|██████████| 48/48 [00:03<00:00, 15.86it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:13] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.745684}
    [2025-04-19 17:21:13] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.745684}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 48/48 [00:03<00:00, 15.96it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:17] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.745684}
    [2025-04-19 17:21:17] [OmniGenome 0.2.4alpha4]  {'root_mean_squared_error': 0.745684}
    
    ---------------------------------------------------- Raw Metric Records ----------------------------------------------------
    ╒═════════════════════════╤═════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                       │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪═════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧═════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    -------------------------------------- https://github.com/yangheng95/metric_visualizer --------------------------------------
    
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-SNMD Progress:  2 / 10 20.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD\\__pycache__\\config.cpython-312.pyc', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD\\__pycache__\\config.cpython-39.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD\config.py>
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-SNMD from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD\config.py
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-SNMD', 'task_type': 'token_classification', 'label2id': {'0': 0, '1': 1}, 'num_labels': 2, 'epochs': 50, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 220, 'patience': 5, 'seeds': [45, 46, 47], 'compute_metrics': <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150FE0>, 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMD/valid.json', 'dataset_cls': <class 'config.Dataset'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>, 'loss_fn': CrossEntropyLoss()}
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:21:18] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-SNMD: task_name: RNA-SNMD
    task_type: token_classification
    label2id: {'0': 0, '1': 1}
    num_labels: 2
    epochs: 1
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 220
    patience: 5
    seeds: [42]
    compute_metrics: <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150FE0>
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/valid.json
    dataset_cls: <class 'config.Dataset'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    loss_fn: CrossEntropyLoss()
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:21:19] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "0",
        "1": "1"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "0": 0,
        "1": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:21:19] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:21:19] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/train.json...
    [2025-04-19 17:21:19] [OmniGenome 0.2.4alpha4]  Loaded 8000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/train.json
    [2025-04-19 17:21:19] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 8000/8000 [00:10<00:00, 786.58it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:29] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 6, 4, 4, 5, 4, 6, 7, 5, 6, 4, 7, 4, 4, 6, 4, 6, 6, 4, 4, 7, 6,
            4, 6, 4, 4, 4, 5, 7, 5, 4, 6, 4, 6, 7, 7, 5, 7, 6, 5, 7, 7, 4, 5, 4, 7,
            5, 5, 4, 6, 5, 4, 6, 6, 7, 6, 4, 5, 4, 5, 7, 5, 5, 5, 4, 5, 7, 7, 5, 4,
            7, 6, 5, 5, 6, 7, 5, 7, 5, 7, 7, 7, 6, 4, 7, 6, 5, 5, 4, 7, 7, 5, 4, 4,
            4, 6, 7, 5, 7, 4, 7, 6, 5, 4, 7, 5, 5, 6, 7, 6, 7, 6, 4, 7, 7, 6, 7, 5,
            4, 6, 5, 6, 4, 5, 4, 5, 5, 4, 7, 4, 6, 4, 6, 5, 7, 5, 5, 5, 6, 7, 7, 7,
            6, 4, 6, 6, 6, 4, 4, 7, 4, 4, 7, 5, 6, 4, 7, 7, 4, 5, 7, 5, 4, 6, 4, 4,
            7, 7, 5, 7, 5, 6, 6, 7, 7, 7, 7, 7, 6, 5, 5, 7, 5, 7, 6, 7, 4, 4, 6, 7,
            6, 7, 7, 6, 5, 4, 7, 7, 4, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    1,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6, 5, 4, 4, 7, 5, 4, 4, 4, 7, 7, 7, 7, 4, 6, 4, 4, 5, 4, 6, 4, 4,
            5, 4, 4, 6, 6, 6, 4, 6, 4, 5, 4, 4, 5, 4, 4, 4, 6, 5, 4, 6, 7, 7, 7, 6,
            4, 7, 7, 4, 7, 7, 5, 7, 6, 6, 7, 6, 6, 5, 6, 7, 4, 7, 4, 6, 4, 7, 4, 4,
            4, 7, 6, 7, 7, 6, 6, 7, 7, 6, 7, 5, 7, 6, 7, 6, 6, 7, 7, 7, 4, 6, 5, 4,
            4, 6, 4, 7, 7, 4, 5, 5, 4, 7, 6, 4, 6, 7, 5, 7, 6, 5, 4, 4, 4, 4, 7, 6,
            7, 4, 7, 5, 4, 4, 4, 6, 7, 5, 7, 4, 6, 5, 4, 4, 7, 6, 5, 5, 5, 7, 7, 6,
            4, 6, 6, 6, 6, 4, 7, 5, 5, 6, 5, 6, 6, 6, 6, 7, 7, 5, 4, 7, 6, 7, 5, 4,
            5, 6, 7, 5, 7, 5, 4, 7, 7, 5, 7, 5, 5, 7, 7, 7, 5, 7, 5, 4, 7, 7, 7, 6,
            4, 6, 4, 5, 6, 4, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    1,    0,    0,    0,    0,    1,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    1,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/test.json...
    [2025-04-19 17:21:30] [OmniGenome 0.2.4alpha4]  Loaded 1000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/test.json
    

.. parsed-literal::

    100%|██████████| 1000/1000 [00:01<00:00, 773.57it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:31] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7, 7, 7, 7, 6, 7, 6, 4, 7, 7, 6, 4, 7, 5, 5, 4, 6, 5, 4, 4, 4, 5,
            4, 4, 6, 7, 7, 6, 7, 5, 7, 6, 5, 7, 4, 4, 7, 7, 4, 5, 4, 7, 6, 6, 7, 6,
            6, 6, 7, 4, 5, 4, 4, 4, 6, 4, 4, 7, 7, 6, 7, 4, 4, 6, 5, 7, 6, 4, 4, 6,
            7, 4, 7, 4, 5, 7, 7, 4, 7, 6, 7, 7, 5, 4, 7, 6, 6, 4, 6, 6, 6, 4, 7, 4,
            6, 5, 7, 4, 5, 4, 7, 7, 7, 6, 4, 6, 5, 5, 7, 7, 6, 7, 7, 4, 7, 6, 4, 5,
            6, 7, 6, 6, 5, 7, 4, 4, 6, 4, 4, 7, 4, 7, 6, 7, 6, 6, 6, 4, 4, 7, 7, 7,
            6, 5, 6, 4, 7, 7, 7, 5, 7, 5, 4, 7, 4, 4, 4, 5, 7, 7, 7, 4, 7, 6, 6, 7,
            6, 6, 7, 4, 4, 7, 4, 6, 7, 5, 7, 5, 4, 4, 6, 6, 5, 4, 4, 5, 7, 7, 4, 7,
            5, 4, 6, 4, 5, 7, 7, 5, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 4, 4, 5, 6, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 7, 6, 4, 7, 6, 4, 7, 6,
            6, 7, 6, 7, 4, 5, 7, 6, 4, 4, 4, 6, 7, 7, 6, 5, 4, 4, 6, 6, 7, 4, 5, 7,
            7, 7, 6, 7, 4, 6, 4, 6, 4, 4, 5, 7, 5, 7, 6, 7, 4, 4, 4, 7, 5, 7, 5, 4,
            4, 6, 4, 4, 4, 5, 7, 7, 6, 5, 7, 5, 7, 4, 5, 4, 7, 7, 7, 6, 5, 4, 7, 7,
            7, 6, 7, 7, 5, 7, 7, 7, 5, 7, 4, 6, 4, 4, 6, 6, 4, 4, 4, 7, 5, 5, 4, 6,
            5, 7, 6, 7, 5, 7, 7, 4, 4, 4, 5, 6, 4, 7, 5, 7, 5, 5, 7, 7, 6, 5, 4, 7,
            7, 6, 5, 5, 6, 4, 6, 4, 5, 6, 4, 7, 5, 7, 4, 6, 5, 4, 7, 6, 7, 6, 7, 5,
            4, 6, 4, 7, 5, 6, 4, 4, 6, 7, 7, 7, 7, 7, 4, 4, 7, 6, 7, 6, 5, 4, 4, 4,
            6, 7, 6, 6, 7, 5, 6, 7, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/valid.json...
    [2025-04-19 17:21:32] [OmniGenome 0.2.4alpha4]  Loaded 1000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMD/valid.json
    

.. parsed-literal::

    100%|██████████| 1000/1000 [00:01<00:00, 790.03it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7, 6, 4, 7, 6, 4, 7, 6, 6, 7, 7, 4, 5, 7, 7, 5, 4, 4, 6, 4, 7, 5,
            4, 6, 6, 4, 6, 4, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 4, 7, 6, 7, 6, 6, 5,
            4, 7, 7, 6, 4, 4, 5, 4, 6, 4, 6, 7, 6, 7, 7, 6, 7, 4, 6, 5, 7, 6, 6, 7,
            7, 7, 4, 5, 5, 7, 7, 5, 4, 6, 4, 6, 4, 4, 6, 4, 4, 5, 6, 7, 4, 7, 7, 7,
            4, 4, 4, 6, 6, 7, 4, 7, 7, 4, 5, 5, 4, 5, 7, 7, 5, 4, 6, 4, 7, 6, 4, 7,
            5, 7, 7, 5, 7, 6, 6, 7, 7, 7, 5, 5, 7, 5, 4, 6, 7, 5, 7, 4, 4, 4, 5, 4,
            4, 6, 4, 5, 6, 4, 7, 6, 7, 6, 7, 5, 5, 5, 7, 4, 5, 7, 7, 5, 7, 6, 7, 4,
            4, 7, 4, 7, 4, 7, 5, 4, 7, 7, 7, 6, 5, 4, 4, 5, 7, 6, 7, 6, 7, 7, 6, 7,
            4, 4, 6, 7, 7, 7, 4, 4, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 5, 5, 5, 7, 7, 4, 4, 4, 6, 5, 4, 5, 4, 7, 5, 6, 7, 7, 6, 7, 4, 4,
            4, 7, 5, 5, 4, 7, 5, 7, 5, 5, 5, 7, 7, 5, 7, 7, 6, 7, 6, 4, 5, 5, 5, 7,
            7, 7, 6, 6, 6, 7, 5, 7, 5, 7, 7, 7, 7, 6, 4, 6, 6, 6, 6, 7, 7, 7, 6, 7,
            7, 6, 7, 4, 7, 5, 6, 6, 4, 4, 5, 5, 4, 7, 6, 7, 7, 4, 5, 4, 4, 4, 7, 5,
            5, 7, 5, 4, 7, 7, 4, 7, 5, 7, 5, 5, 6, 4, 6, 6, 7, 6, 7, 4, 7, 4, 4, 4,
            5, 4, 7, 4, 4, 4, 7, 7, 7, 4, 7, 5, 6, 4, 4, 5, 7, 5, 6, 5, 4, 4, 7, 7,
            7, 7, 5, 4, 6, 4, 7, 7, 7, 7, 6, 7, 4, 5, 7, 7, 4, 4, 4, 4, 6, 4, 4, 7,
            6, 6, 7, 7, 7, 5, 4, 7, 7, 5, 6, 7, 7, 6, 4, 6, 4, 7, 7, 4, 4, 7, 7, 7,
            7, 4, 6, 4, 5, 5, 7, 7, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:21:33] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 250/250 [00:15<00:00, 15.95it/s]
    

.. parsed-literal::

    [2025-04-19 17:21:49] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.4817431275553723}
    [2025-04-19 17:21:49] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.4817431275553723}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.6202: 100%|██████████| 2000/2000 [05:43<00:00,  5.83it/s]
    Evaluating: 100%|██████████| 250/250 [00:15<00:00, 16.34it/s]
    

.. parsed-literal::

    [2025-04-19 17:27:48] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.5957607807439937}
    [2025-04-19 17:27:48] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.5957607807439937}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 250/250 [00:15<00:00, 16.26it/s]
    

.. parsed-literal::

    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.5970559803186319}
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  {'roc_auc_score': 0.5970559803186319}
    
    ---------------------------------------------------- Raw Metric Records ----------------------------------------------------
    ╒═════════════════════════╤═════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                       │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪═════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼─────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧═════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    -------------------------------------- https://github.com/yangheng95/metric_visualizer --------------------------------------
    
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-SNMR Progress:  3 / 10 30.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMR\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMR\\__pycache__\\config.cpython-312.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR\config.py>
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-SNMR from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR\config.py
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-SNMR', 'task_type': 'token_classification', 'label2id': {'A': 0, 'T': 1, 'G': 2, 'C': 3}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 220, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7A22272E0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA948860>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMR/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMR/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SNMR/valid.json', 'dataset_cls': <class 'config.Dataset'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:28:05] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-SNMR: task_name: RNA-SNMR
    task_type: token_classification
    label2id: {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 220
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7A22272E0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA948860>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/valid.json
    dataset_cls: <class 'config.Dataset'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:28:10] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "A",
        "1": "T",
        "2": "G",
        "3": "C"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "A": 0,
        "C": 3,
        "G": 2,
        "T": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:28:10] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:28:10] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/train.json...
    [2025-04-19 17:28:10] [OmniGenome 0.2.4alpha4]  Loaded 8000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/train.json
    [2025-04-19 17:28:10] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 8000/8000 [00:10<00:00, 795.02it/s]
    

.. parsed-literal::

    [2025-04-19 17:28:20] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 6, 4, 4, 5, 4, 6, 7, 5, 6, 4, 7, 4, 4, 6, 4, 6, 6, 4, 4, 7, 6,
            4, 6, 4, 4, 4, 5, 7, 5, 4, 6, 4, 6, 7, 7, 5, 7, 6, 5, 7, 7, 4, 5, 4, 7,
            5, 5, 4, 6, 5, 4, 6, 6, 7, 6, 4, 5, 4, 5, 7, 5, 5, 5, 4, 5, 7, 7, 5, 4,
            7, 6, 5, 5, 6, 7, 5, 7, 5, 7, 7, 7, 6, 4, 7, 6, 5, 5, 4, 7, 7, 5, 4, 4,
            4, 6, 7, 5, 7, 4, 7, 6, 5, 4, 7, 5, 5, 6, 7, 6, 7, 6, 4, 7, 7, 6, 7, 5,
            4, 6, 5, 6, 4, 5, 4, 5, 5, 4, 7, 4, 6, 4, 6, 5, 7, 5, 5, 5, 6, 7, 7, 7,
            6, 4, 6, 6, 6, 4, 4, 7, 4, 4, 7, 5, 6, 4, 7, 7, 4, 5, 7, 5, 4, 6, 4, 4,
            7, 7, 5, 7, 5, 6, 6, 7, 7, 7, 7, 7, 6, 5, 5, 7, 5, 7, 6, 7, 4, 4, 6, 7,
            6, 7, 7, 6, 5, 4, 7, 7, 4, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100,    1, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100,    3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100,    0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6, 5, 4, 4, 7, 5, 4, 4, 4, 7, 7, 7, 7, 4, 6, 4, 4, 5, 4, 6, 4, 4,
            5, 4, 4, 6, 6, 6, 4, 6, 4, 5, 4, 4, 5, 4, 4, 4, 6, 5, 4, 6, 7, 7, 7, 6,
            4, 7, 7, 4, 7, 7, 5, 7, 6, 6, 7, 6, 6, 5, 6, 7, 4, 7, 4, 6, 4, 7, 4, 4,
            4, 7, 6, 7, 7, 6, 6, 7, 7, 6, 7, 5, 7, 6, 7, 6, 6, 7, 7, 7, 4, 6, 5, 4,
            4, 6, 4, 7, 7, 4, 5, 5, 4, 7, 6, 4, 6, 7, 5, 7, 6, 5, 4, 4, 4, 4, 7, 6,
            7, 4, 7, 5, 4, 4, 4, 6, 7, 5, 7, 4, 6, 5, 4, 4, 7, 6, 5, 5, 5, 7, 7, 6,
            4, 6, 6, 6, 6, 4, 7, 5, 5, 6, 5, 6, 6, 6, 6, 7, 7, 5, 4, 7, 6, 7, 5, 4,
            5, 6, 7, 5, 7, 5, 4, 7, 7, 5, 7, 5, 5, 7, 7, 7, 5, 7, 5, 4, 7, 7, 7, 6,
            4, 6, 4, 5, 6, 4, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100,    1, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,    2, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100,    1, -100, -100, -100, -100,    0, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100,    0, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/test.json...
    [2025-04-19 17:28:21] [OmniGenome 0.2.4alpha4]  Loaded 1000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/test.json
    

.. parsed-literal::

    100%|██████████| 1000/1000 [00:01<00:00, 795.88it/s]
    

.. parsed-literal::

    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7, 7, 7, 7, 6, 7, 6, 4, 7, 7, 6, 4, 7, 5, 5, 4, 6, 5, 4, 4, 4, 5,
            4, 4, 6, 7, 7, 6, 7, 5, 7, 6, 5, 7, 4, 4, 7, 7, 4, 5, 4, 7, 6, 6, 7, 6,
            6, 6, 7, 4, 5, 4, 4, 4, 6, 4, 4, 7, 7, 6, 7, 4, 4, 6, 5, 7, 6, 4, 4, 6,
            7, 4, 7, 4, 5, 7, 7, 4, 7, 6, 7, 7, 5, 4, 7, 6, 6, 4, 6, 6, 6, 4, 7, 4,
            6, 5, 7, 4, 5, 4, 7, 7, 7, 6, 4, 6, 5, 5, 7, 7, 6, 7, 7, 4, 7, 6, 4, 5,
            6, 7, 6, 6, 5, 7, 4, 4, 6, 4, 4, 7, 4, 7, 6, 7, 6, 6, 6, 4, 4, 7, 7, 7,
            6, 5, 6, 4, 7, 7, 7, 5, 7, 5, 4, 7, 4, 4, 4, 5, 7, 7, 7, 4, 7, 6, 6, 7,
            6, 6, 7, 4, 4, 7, 4, 6, 7, 5, 7, 5, 4, 4, 6, 6, 5, 4, 4, 5, 7, 7, 4, 7,
            5, 4, 6, 4, 5, 7, 7, 5, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100,    3, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100,    2, -100, -100, -100,    0, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 4, 4, 5, 6, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 7, 6, 4, 7, 6, 4, 7, 6,
            6, 7, 6, 7, 4, 5, 7, 6, 4, 4, 4, 6, 7, 7, 6, 5, 4, 4, 6, 6, 7, 4, 5, 7,
            7, 7, 6, 7, 4, 6, 4, 6, 4, 4, 5, 7, 5, 7, 6, 7, 4, 4, 4, 7, 5, 7, 5, 4,
            4, 6, 4, 4, 4, 5, 7, 7, 6, 5, 7, 5, 7, 4, 5, 4, 7, 7, 7, 6, 5, 4, 7, 7,
            7, 6, 7, 7, 5, 7, 7, 7, 5, 7, 4, 6, 4, 4, 6, 6, 4, 4, 4, 7, 5, 5, 4, 6,
            5, 7, 6, 7, 5, 7, 7, 4, 4, 4, 5, 6, 4, 7, 5, 7, 5, 5, 7, 7, 6, 5, 4, 7,
            7, 6, 5, 5, 6, 4, 6, 4, 5, 6, 4, 7, 5, 7, 4, 6, 5, 4, 7, 6, 7, 6, 7, 5,
            4, 6, 4, 7, 5, 6, 4, 4, 6, 7, 7, 7, 7, 7, 4, 4, 7, 6, 7, 6, 5, 4, 4, 4,
            6, 7, 6, 6, 7, 5, 6, 7, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100,    1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  Detected max_length=220 in the dataset, using it as the max_length.
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/valid.json...
    [2025-04-19 17:28:22] [OmniGenome 0.2.4alpha4]  Loaded 1000 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SNMR/valid.json
    

.. parsed-literal::

    100%|██████████| 1000/1000 [00:01<00:00, 784.98it/s]
    

.. parsed-literal::

    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=208, label_padding_length=208
    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 202.0, 'max_seq_len': 202, 'min_seq_len': 202, 'avg_label_len': 208.0, 'max_label_len': 208, 'min_label_len': 208}
    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7, 6, 4, 7, 6, 4, 7, 6, 6, 7, 7, 4, 5, 7, 7, 5, 4, 4, 6, 4, 7, 5,
            4, 6, 6, 4, 6, 4, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 4, 7, 6, 7, 6, 6, 5,
            4, 7, 7, 6, 4, 4, 5, 4, 6, 4, 6, 7, 6, 7, 7, 6, 7, 4, 6, 5, 7, 6, 6, 7,
            7, 7, 4, 5, 5, 7, 7, 5, 4, 6, 4, 6, 4, 4, 6, 4, 4, 5, 6, 7, 4, 7, 7, 7,
            4, 4, 4, 6, 6, 7, 4, 7, 7, 4, 5, 5, 4, 5, 7, 7, 5, 4, 6, 4, 7, 6, 4, 7,
            5, 7, 7, 5, 7, 6, 6, 7, 7, 7, 5, 5, 7, 5, 4, 6, 7, 5, 7, 4, 4, 4, 5, 4,
            4, 6, 4, 5, 6, 4, 7, 6, 7, 6, 7, 5, 5, 5, 7, 4, 5, 7, 7, 5, 7, 6, 7, 4,
            4, 7, 4, 7, 4, 7, 5, 4, 7, 7, 7, 6, 5, 4, 4, 5, 7, 6, 7, 6, 7, 7, 6, 7,
            4, 4, 6, 7, 7, 7, 4, 4, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 5, 5, 5, 7, 7, 4, 4, 4, 6, 5, 4, 5, 4, 7, 5, 6, 7, 7, 6, 7, 4, 4,
            4, 7, 5, 5, 4, 7, 5, 7, 5, 5, 5, 7, 7, 5, 7, 7, 6, 7, 6, 4, 5, 5, 5, 7,
            7, 7, 6, 6, 6, 7, 5, 7, 5, 7, 7, 7, 7, 6, 4, 6, 6, 6, 6, 7, 7, 7, 6, 7,
            7, 6, 7, 4, 7, 5, 6, 6, 4, 4, 5, 5, 4, 7, 6, 7, 7, 4, 5, 4, 4, 4, 7, 5,
            5, 7, 5, 4, 7, 7, 4, 7, 5, 7, 5, 5, 6, 4, 6, 6, 7, 6, 7, 4, 7, 4, 4, 4,
            5, 4, 7, 4, 4, 4, 7, 7, 7, 4, 7, 5, 6, 4, 4, 5, 7, 5, 6, 5, 4, 4, 7, 7,
            7, 7, 5, 4, 6, 4, 7, 7, 7, 7, 6, 7, 4, 5, 7, 7, 4, 4, 4, 4, 6, 4, 4, 7,
            6, 6, 7, 7, 7, 5, 4, 7, 7, 5, 6, 7, 7, 6, 4, 6, 4, 7, 7, 4, 4, 7, 7, 7,
            7, 4, 6, 4, 5, 5, 7, 7, 7, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:28:24] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 250/250 [00:15<00:00, 16.21it/s]
    

.. parsed-literal::

    [2025-04-19 17:28:40] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.2737850675253224, 'matthews_corrcoef': 0.06335346758418496}
    [2025-04-19 17:28:40] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.2737850675253224, 'matthews_corrcoef': 0.06335346758418496}
    

.. parsed-literal::

    Epoch 1/1 Loss: 1.2502: 100%|██████████| 2000/2000 [05:43<00:00,  5.82it/s]
    Evaluating: 100%|██████████| 250/250 [00:15<00:00, 16.47it/s]
    

.. parsed-literal::

    [2025-04-19 17:34:39] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.44408230644413554, 'matthews_corrcoef': 0.2776818514600447}
    [2025-04-19 17:34:39] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.44408230644413554, 'matthews_corrcoef': 0.2776818514600447}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 250/250 [00:15<00:00, 16.49it/s]
    

.. parsed-literal::

    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.4705120824131688, 'matthews_corrcoef': 0.3114356674710262}
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.4705120824131688, 'matthews_corrcoef': 0.3114356674710262}
    
    ---------------------------------------------------- Raw Metric Records ----------------------------------------------------
    ╒═════════════════════════╤═════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                       │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪═════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼─────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼─────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼─────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧═════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    -------------------------------------- https://github.com/yangheng95/metric_visualizer --------------------------------------
    
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-SSP-Archive2 Progress:  4 / 10 40.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2\\__pycache__\\config.cpython-312.pyc', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2\\__pycache__\\config.cpython-39.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2\config.py>
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-SSP-Archive2 from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2\config.py
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-SSP-Archive2', 'task_type': 'token_classification', 'label2id': {'(': 0, ')': 1, '.': 2}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA9484A0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150EA0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-Archive2/valid.json', 'dataset_cls': <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:34:55] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:34:56] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:34:56] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-SSP-Archive2: task_name: RNA-SSP-Archive2
    task_type: token_classification
    label2id: {'(': 0, ')': 1, '.': 2}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA9484A0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150EA0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/valid.json
    dataset_cls: <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/train.json...
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Loaded 608 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/train.json
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 608/608 [00:00<00:00, 1031.73it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=504
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 130.54276315789474, 'max_seq_len': 501, 'min_seq_len': 56, 'avg_label_len': 504.0, 'max_label_len': 504, 'min_label_len': 504}
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 6, 5, 5, 5, 5, 9, 4, 6, 5, 9, 5, 4, 6, 9, 5, 9, 6, 6, 9, 5, 4,
            6, 4, 6, 5, 6, 5, 9, 5, 6, 6, 5, 9, 9, 4, 9, 4, 4, 5, 5, 6, 6, 6, 9, 6,
            6, 9, 5, 4, 9, 6, 6, 6, 9, 9, 5, 6, 4, 4, 5, 5, 5, 5, 4, 9, 6, 6, 6, 6,
            5, 5, 5, 4, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,
               0,    0,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
               1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    2,    2,    2,    1,    1,    1,    1,    1,    1,    2,
               2,    2,    2,    0,    0,    0,    0,    0,    2,    2,    2,    2,
               2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    1,    1,    2,    2,    2,    2, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 5, 5, 5, 4, 9, 6, 9, 4, 6, 5, 9, 5, 4, 6, 9, 4, 6, 6, 4, 9, 4, 6,
            4, 6, 5, 4, 5, 6, 5, 6, 5, 5, 9, 9, 5, 9, 4, 4, 6, 5, 6, 9, 6, 4, 6, 6,
            9, 5, 6, 6, 4, 4, 6, 9, 9, 5, 6, 4, 6, 5, 5, 9, 9, 5, 9, 5, 6, 9, 6, 6,
            6, 5, 4, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,
               0,    0,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,
               1,    1,    1,    2,    0,    0,    0,    0,    0,    2,    2,    2,
               2,    2,    2,    2,    1,    1,    1,    1,    1,    2,    2,    2,
               2,    2,    0,    0,    0,    0,    0,    2,    2,    2,    2,    2,
               2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    1,    2,    2,    2,    2, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/test.json...
    [2025-04-19 17:35:00] [OmniGenome 0.2.4alpha4]  Loaded 82 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/test.json
    

.. parsed-literal::

    100%|██████████| 82/82 [00:00<00:00, 1047.57it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=328, label_padding_length=328
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 131.23170731707316, 'max_seq_len': 321, 'min_seq_len': 67, 'avg_label_len': 328.0, 'max_label_len': 328, 'min_label_len': 328}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 9, 6, 5, 6, 5, 6, 6, 9, 4, 6, 6, 4, 6, 4, 6, 9, 6, 6, 4, 4, 5, 9, 5,
            5, 6, 4, 5, 6, 6, 6, 5, 9, 5, 4, 9, 4, 4, 5, 5, 5, 6, 9, 4, 6, 6, 9, 5,
            5, 5, 4, 6, 6, 4, 9, 5, 6, 4, 4, 4, 5, 5, 9, 6, 6, 5, 5, 6, 5, 6, 5, 4,
            4, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,
               0,    0,    2,    2,    2,    2,    2,    2,    2,    1,    1,    1,
               1,    2,    0,    0,    0,    0,    0,    2,    2,    2,    2,    2,
               2,    2,    1,    1,    1,    1,    1,    2,    2,    2,    2,    2,
               0,    0,    0,    0,    0,    2,    2,    2,    2,    2,    2,    2,
               1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               2,    2,    2,    2, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 4, 6, 9, 4, 5, 6, 4, 5, 5, 4, 9, 4, 5, 9, 9, 6, 4, 6, 9, 6, 4, 4,
            4, 4, 5, 4, 5, 5, 4, 9, 4, 9, 5, 5, 5, 6, 9, 5, 5, 6, 4, 9, 9, 9, 6, 9,
            6, 4, 4, 6, 9, 9, 4, 4, 6, 5, 4, 5, 5, 5, 4, 5, 4, 6, 6, 5, 9, 9, 4, 6,
            9, 9, 4, 6, 9, 4, 5, 9, 6, 4, 6, 6, 9, 5, 4, 6, 9, 6, 4, 9, 6, 4, 5, 9,
            5, 6, 6, 6, 4, 4, 5, 5, 5, 9, 6, 4, 6, 9, 6, 5, 5, 6, 9, 4, 5, 9, 5, 5,
            5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    0,    2,    2,
               2,    2,    0,    0,    0,    0,    2,    0,    0,    0,    2,    2,
               2,    2,    2,    0,    0,    0,    0,    0,    0,    2,    2,    2,
               2,    2,    2,    2,    2,    2,    2,    2,    2,    1,    1,    1,
               1,    2,    2,    1,    1,    2,    2,    2,    2,    1,    1,    1,
               2,    1,    1,    2,    1,    1,    2,    0,    0,    0,    0,    0,
               2,    2,    2,    2,    2,    2,    0,    0,    0,    0,    0,    2,
               0,    0,    0,    2,    2,    2,    2,    1,    1,    1,    1,    1,
               1,    1,    1,    2,    2,    2,    2,    2,    1,    1,    1,    1,
               1,    2,    1,    2,    1,    1,    1,    1,    1,    1,    1,    2,
               2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100])}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/valid.json...
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Loaded 76 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-Archive2/valid.json
    

.. parsed-literal::

    100%|██████████| 76/76 [00:00<00:00, 1109.12it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=312, label_padding_length=312
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 117.39473684210526, 'max_seq_len': 308, 'min_seq_len': 60, 'avg_label_len': 312.0, 'max_label_len': 312, 'min_label_len': 312}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 9, 9, 5, 5, 6, 6, 5, 6, 4, 9, 5, 4, 9, 4, 9, 5, 9, 9, 4, 4, 4, 6, 6,
            9, 9, 4, 9, 4, 5, 5, 9, 6, 9, 9, 5, 5, 5, 4, 9, 9, 5, 5, 6, 4, 4, 5, 4,
            5, 4, 6, 5, 4, 6, 9, 5, 4, 4, 6, 5, 9, 9, 9, 4, 4, 6, 4, 6, 5, 5, 6, 4,
            9, 6, 4, 9, 4, 6, 9, 6, 5, 5, 5, 4, 5, 5, 4, 6, 5, 6, 9, 6, 4, 4, 4, 6,
            9, 4, 6, 6, 9, 5, 9, 9, 6, 5, 5, 6, 6, 4, 9, 5, 2, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    2,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    2,    0,    0,    0,    0,    0,    0,    0,    0,    2,
               2,    2,    2,    2,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,
               1,    1,    1,    2,    2,    1,    1,    2,    2,    2,    2,    1,
               1,    1,    1,    1,    1,    1,    1,    2,    0,    0,    2,    0,
               0,    2,    2,    2,    2,    0,    0,    0,    0,    2,    2,    2,
               2,    2,    2,    1,    1,    1,    1,    2,    2,    2,    2,    1,
               1,    2,    1,    1,    2,    2,    1,    1,    1,    1,    1,    1,
               1,    1,    2,    2, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 6, 5, 5, 5, 5, 5, 4, 6, 5, 5, 4, 9, 6, 6, 6, 6, 6, 5, 5, 5, 5, 6,
            4, 4, 6, 6, 5, 6, 5, 5, 4, 5, 6, 5, 6, 6, 6, 6, 6, 5, 5, 4, 6, 4, 4, 6,
            4, 4, 5, 5, 5, 6, 5, 5, 5, 5, 5, 4, 6, 4, 6, 5, 9, 9, 6, 4, 4, 6, 5, 4,
            6, 6, 5, 5, 4, 6, 5, 4, 4, 6, 6, 6, 5, 5, 9, 6, 5, 4, 6, 6, 6, 9, 4, 5,
            4, 4, 9, 6, 4, 4, 5, 5, 5, 5, 6, 9, 5, 5, 5, 6, 5, 6, 6, 6, 6, 5, 5, 6,
            6, 5, 6, 6, 4, 6, 6, 5, 6, 6, 6, 5, 6, 6, 5, 5, 4, 6, 5, 5, 6, 6, 4, 6,
            6, 6, 5, 5, 6, 6, 5, 5, 6, 4, 4, 6, 5, 5, 6, 5, 5, 6, 9, 4, 6, 5, 5, 6,
            6, 6, 6, 5, 5, 4, 5, 5, 5, 6, 6, 5, 6, 4, 6, 6, 5, 5, 5, 6, 6, 4, 4, 6,
            6, 6, 4, 6, 5, 4, 6, 5, 5, 6, 4, 5, 5, 5, 5, 6, 6, 5, 5, 6, 4, 5, 5, 6,
            6, 5, 6, 9, 9, 5, 6, 5, 6, 6, 6, 6, 6, 6, 4, 4, 5, 6, 6, 6, 6, 6, 6, 4,
            6, 4, 4, 6, 5, 5, 5, 9, 6, 5, 4, 6, 6, 6, 9, 4, 4, 5, 5, 5, 9, 5, 9, 6,
            6, 5, 5, 9, 6, 5, 9, 9, 5, 4, 4, 4, 5, 9, 5, 9, 6, 6, 6, 4, 6, 5, 6, 6,
            6, 6, 6, 5, 9, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 5, 5, 2, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,
               2,    2,    2,    0,    0,    0,    2,    2,    2,    1,    1,    1,
               2,    1,    1,    1,    1,    1,    1,    2,    2,    2,    2,    2,
               2,    2,    1,    1,    1,    1,    2,    0,    0,    0,    2,    2,
               2,    2,    2,    0,    0,    0,    0,    0,    2,    2,    2,    0,
               0,    0,    0,    0,    0,    0,    2,    2,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    0,    2,    2,    2,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,
               0,    0,    0,    0,    2,    0,    0,    0,    0,    0,    2,    0,
               0,    0,    0,    0,    2,    0,    0,    0,    2,    2,    2,    2,
               1,    1,    1,    2,    1,    1,    1,    1,    1,    2,    2,    2,
               1,    1,    1,    1,    1,    2,    1,    2,    2,    0,    0,    0,
               0,    0,    0,    2,    2,    2,    2,    2,    0,    0,    0,    0,
               2,    2,    2,    2,    0,    0,    0,    2,    2,    2,    2,    1,
               1,    1,    2,    2,    2,    2,    1,    1,    1,    1,    2,    1,
               1,    1,    1,    1,    1,    2,    1,    1,    2,    1,    1,    1,
               1,    1,    2,    1,    2,    1,    1,    1,    1,    1,    1,    2,
               2,    2,    2,    1,    1,    1,    1,    1,    1,    2,    1,    2,
               2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    2,    2,    2,    2,    1,    1,    1,    1,    1,    1,    1,
               1,    1,    1,    1,    2,    2,    1,    2,    1,    1,    1,    2,
               2,    2,    2,    2,    1,    2,    2,    2,    2,    2,    1,    1,
               1,    1,    1,    1,    1,    2,    1,    1,    1,    1,    1,    1,
               1,    2,    2,    2,    2, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:01] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 19/19 [00:01<00:00, 15.49it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:02] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.4720883878727751, 'matthews_corrcoef': 0.23668201635681113}
    [2025-04-19 17:35:02] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.4720883878727751, 'matthews_corrcoef': 0.23668201635681113}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.7500: 100%|██████████| 152/152 [00:25<00:00,  5.85it/s]
    Evaluating: 100%|██████████| 19/19 [00:01<00:00, 15.53it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:30] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.8876411792666348, 'matthews_corrcoef': 0.8244443180430253}
    [2025-04-19 17:35:30] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.8876411792666348, 'matthews_corrcoef': 0.8244443180430253}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 21/21 [00:01<00:00, 15.45it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:32] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.8800914439102971, 'matthews_corrcoef': 0.813313796203534}
    [2025-04-19 17:35:32] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.8800914439102971, 'matthews_corrcoef': 0.813313796203534}
    
    -------------------------------------------------------- Raw Metric Records --------------------------------------------------------
    ╒═════════════════════════╤═════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                               │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪═════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M         │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼─────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼─────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M         │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼─────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼─────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M         │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼─────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M         │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧═════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ------------------------------------------ https://github.com/yangheng95/metric_visualizer ------------------------------------------
    
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-SSP-rnastralign Progress:  5 / 10 50.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-rnastralign\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-rnastralign\\__pycache__\\config.cpython-312.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign\config.py>
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-SSP-rnastralign from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign\config.py
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-SSP-rnastralign', 'task_type': 'token_classification', 'label2id': {'(': 0, ')': 1, '.': 2}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA8A1080>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA927560>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-rnastralign/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-rnastralign/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-rnastralign/valid.json', 'dataset_cls': <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:35:33] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-SSP-rnastralign: task_name: RNA-SSP-rnastralign
    task_type: token_classification
    label2id: {'(': 0, ')': 1, '.': 2}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA8A1080>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA927560>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/valid.json
    dataset_cls: <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:35:36] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:35:36] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:36] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/train.json...
    [2025-04-19 17:35:36] [OmniGenome 0.2.4alpha4]  Loaded 3104 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/train.json
    [2025-04-19 17:35:36] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 3104/3104 [00:02<00:00, 1321.78it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=504
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 101.23228092783505, 'max_seq_len': 501, 'min_seq_len': 68, 'avg_label_len': 504.0, 'max_label_len': 504, 'min_label_len': 504}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 9, 9, 9, 5, 6, 6, 9, 5, 9, 9, 9, 4, 6, 9, 6, 5, 6, 4, 9, 6, 9, 6,
            6, 4, 4, 5, 5, 4, 5, 9, 4, 5, 9, 4, 9, 5, 5, 4, 9, 9, 5, 5, 6, 4, 4, 5,
            4, 6, 4, 9, 4, 4, 6, 9, 6, 4, 4, 4, 5, 4, 5, 4, 9, 5, 4, 6, 5, 6, 5, 9,
            6, 4, 5, 6, 4, 9, 4, 6, 9, 9, 6, 4, 5, 9, 5, 6, 5, 4, 4, 6, 6, 6, 9, 5,
            5, 6, 5, 6, 4, 4, 4, 4, 9, 4, 6, 6, 9, 5, 4, 4, 6, 6, 5, 5, 6, 6, 4, 6,
            6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,
               2,    2,    2,    2,    0,    0,    0,    0,    0,    0,    0,    0,
               2,    2,    2,    2,    2,    0,    0,    2,    2,    0,    0,    2,
               2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
               1,    1,    2,    2,    2,    2,    1,    1,    2,    2,    2,    2,
               1,    1,    1,    1,    1,    1,    2,    1,    1,    2,    0,    0,
               2,    0,    2,    2,    2,    2,    2,    0,    0,    2,    0,    0,
               0,    0,    0,    2,    2,    2,    2,    1,    1,    1,    1,    1,
               2,    1,    1,    2,    2,    2,    2,    2,    1,    2,    1,    1,
               2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 9, 6, 5, 6, 5, 9, 9, 6, 5, 6, 5, 9, 5, 6, 6, 9, 6, 9, 6, 9, 4, 6, 9,
            6, 9, 4, 5, 4, 9, 4, 5, 5, 5, 6, 4, 9, 5, 5, 5, 4, 9, 5, 5, 5, 6, 4, 6,
            5, 9, 5, 6, 6, 5, 5, 6, 9, 6, 4, 4, 4, 5, 4, 5, 9, 5, 5, 4, 6, 5, 6, 5,
            5, 9, 4, 9, 9, 6, 9, 4, 5, 5, 6, 4, 6, 6, 5, 9, 9, 4, 4, 6, 5, 6, 6, 5,
            6, 6, 6, 4, 6, 4, 6, 9, 5, 6, 6, 9, 5, 4, 6, 5, 6, 5, 5, 4, 6, 6, 6, 5,
            5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    2,    2,    2,    2,    0,    0,    0,    0,    0,    0,    0,
               2,    2,    2,    2,    2,    0,    0,    0,    2,    0,    0,    0,
               0,    2,    2,    2,    2,    2,    0,    0,    0,    0,    0,    0,
               2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
               2,    1,    1,    1,    1,    2,    2,    1,    1,    2,    2,    2,
               2,    1,    1,    1,    1,    2,    1,    2,    1,    1,    2,    0,
               0,    2,    0,    0,    2,    2,    2,    2,    0,    0,    0,    2,
               2,    0,    0,    2,    2,    2,    2,    1,    1,    2,    2,    1,
               1,    1,    2,    2,    2,    2,    1,    1,    2,    1,    1,    2,
               2,    2,    1,    1,    1,    1,    2,    1,    1,    1,    2,    2,
               2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/test.json...
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Loaded 389 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/test.json
    

.. parsed-literal::

    100%|██████████| 389/389 [00:00<00:00, 1329.99it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=392, label_padding_length=392
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 95.96401028277634, 'max_seq_len': 388, 'min_seq_len': 55, 'avg_label_len': 392.0, 'max_label_len': 392, 'min_label_len': 392}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 9, 5, 5, 9, 6, 6, 9, 6, 6, 5, 9, 4, 9, 4, 6, 5, 6, 4, 6, 6, 6, 6, 6,
            4, 4, 4, 5, 6, 5, 5, 9, 6, 4, 9, 5, 5, 5, 4, 9, 9, 5, 5, 6, 4, 4, 5, 9,
            5, 4, 6, 4, 4, 6, 5, 9, 4, 4, 6, 5, 5, 5, 5, 9, 5, 5, 4, 5, 6, 5, 5, 6,
            4, 9, 6, 6, 9, 4, 5, 9, 6, 5, 6, 6, 9, 6, 9, 9, 9, 5, 6, 5, 5, 6, 9, 6,
            6, 6, 4, 6, 4, 6, 9, 4, 6, 6, 9, 5, 4, 5, 9, 6, 5, 5, 4, 6, 6, 4, 2, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    2,    0,    2,    0,    0,    0,    0,    0,    0,    2,
               2,    2,    2,    2,    0,    0,    0,    0,    0,    0,    2,    2,
               2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,
               1,    1,    1,    2,    2,    1,    1,    2,    2,    2,    2,    1,
               1,    1,    1,    1,    1,    2,    2,    1,    2,    0,    0,    2,
               0,    0,    2,    2,    2,    2,    0,    0,    0,    0,    0,    0,
               0,    0,    2,    2,    2,    1,    1,    1,    1,    1,    1,    1,
               1,    2,    2,    2,    2,    1,    1,    2,    1,    1,    2,    2,
               2,    1,    1,    1,    1,    1,    1,    1,    1,    1, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 6, 4, 9, 6, 9, 6, 6, 5, 4, 6, 4, 6, 5, 6, 6, 5, 5, 6, 4, 4,
            9, 6, 5, 4, 5, 9, 6, 6, 9, 5, 9, 9, 6, 4, 4, 4, 4, 5, 5, 4, 6, 5, 6, 4,
            9, 6, 6, 6, 4, 4, 4, 5, 5, 4, 9, 5, 5, 5, 4, 6, 6, 6, 9, 9, 5, 4, 4, 4,
            9, 5, 5, 5, 9, 6, 5, 6, 9, 5, 9, 5, 5, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,
               0,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
               1,    1,    1,    2,    0,    0,    0,    0,    0,    2,    2,    2,
               2,    2,    2,    2,    1,    1,    1,    1,    1,    2,    2,    0,
               0,    0,    0,    2,    2,    2,    2,    1,    1,    1,    1,    2,
               2,    0,    0,    0,    0,    0,    2,    2,    2,    2,    2,    2,
               2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/valid.json...
    [2025-04-19 17:35:39] [OmniGenome 0.2.4alpha4]  Loaded 388 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-rnastralign/valid.json
    

.. parsed-literal::

    100%|██████████| 388/388 [00:00<00:00, 1228.98it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=480, label_padding_length=480
    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 106.70618556701031, 'max_seq_len': 477, 'min_seq_len': 72, 'avg_label_len': 480.0, 'max_label_len': 480, 'min_label_len': 480}
    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 6, 4, 6, 6, 5, 6, 9, 6, 9, 5, 5, 6, 4, 4, 5, 9, 6, 6, 5, 9, 4, 4,
            6, 6, 4, 6, 5, 5, 6, 6, 9, 5, 9, 9, 6, 4, 4, 4, 4, 5, 5, 6, 6, 9, 6, 6,
            6, 5, 5, 6, 6, 4, 4, 6, 6, 5, 5, 5, 6, 9, 6, 9, 6, 6, 6, 9, 9, 5, 6, 4,
            6, 9, 5, 5, 5, 4, 5, 5, 6, 5, 5, 9, 5, 5, 6, 5, 5, 4, 2, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,
               0,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
               1,    1,    1,    0,    0,    0,    0,    0,    0,    2,    2,    2,
               2,    2,    2,    2,    1,    1,    1,    1,    1,    1,    0,    0,
               0,    0,    0,    2,    2,    2,    2,    1,    1,    1,    1,    1,
               2,    2,    0,    0,    0,    0,    0,    2,    2,    2,    2,    2,
               2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
               1,    1,    2,    2,    2,    2, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6, 5, 5, 9, 9, 9, 9, 5, 9, 9, 4, 6, 4, 5, 5, 9, 6, 9, 6, 5, 4, 4,
            9, 6, 5, 9, 4, 9, 9, 9, 9, 9, 4, 4, 6, 5, 4, 5, 5, 6, 5, 9, 9, 5, 4, 6,
            6, 6, 9, 6, 6, 6, 4, 4, 5, 4, 5, 4, 6, 5, 4, 6, 4, 6, 5, 4, 5, 9, 9, 6,
            4, 9, 9, 9, 9, 4, 6, 9, 6, 9, 6, 9, 6, 5, 5, 6, 5, 4, 6, 9, 9, 4, 9, 5,
            9, 6, 6, 6, 6, 4, 4, 6, 6, 6, 9, 9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor([-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               2,    0,    2,    2,    0,    0,    0,    0,    0,    0,    2,    0,
               0,    0,    0,    0,    0,    2,    2,    2,    0,    0,    0,    0,
               0,    2,    2,    2,    2,    0,    0,    0,    0,    2,    2,    2,
               2,    0,    0,    0,    2,    2,    2,    2,    1,    1,    1,    2,
               2,    2,    2,    1,    1,    1,    1,    2,    1,    1,    1,    1,
               1,    2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    2,
               2,    2,    1,    1,    1,    1,    1,    2,    2,    2,    2,    2,
               1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])}
    [2025-04-19 17:35:40] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 97/97 [00:06<00:00, 15.76it/s]
    

.. parsed-literal::

    [2025-04-19 17:35:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.48003478513947645, 'matthews_corrcoef': 0.251113820008015}
    [2025-04-19 17:35:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.48003478513947645, 'matthews_corrcoef': 0.251113820008015}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.6091: 100%|██████████| 776/776 [02:13<00:00,  5.82it/s]
    Evaluating: 100%|██████████| 97/97 [00:06<00:00, 16.06it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:06] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9373069106292483, 'matthews_corrcoef': 0.9028343575144284}
    [2025-04-19 17:38:06] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9373069106292483, 'matthews_corrcoef': 0.9028343575144284}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 98/98 [00:06<00:00, 16.21it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:13] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9728583458467837, 'matthews_corrcoef': 0.9579414610773724}
    [2025-04-19 17:38:13] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9728583458467837, 'matthews_corrcoef': 0.9579414610773724}
    
    ---------------------------------------------------------- Raw Metric Records ----------------------------------------------------------
    ╒═════════════════════════╤════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                  │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M            │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M    │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M            │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M    │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M            │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M            │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ------------------------------------------- https://github.com/yangheng95/metric_visualizer -------------------------------------------
    
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-SSP-bpRNA Progress:  6 / 10 60.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-bpRNA\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-bpRNA\\__pycache__\\config.cpython-312.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA\config.py>
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-SSP-bpRNA from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA\config.py
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-SSP-bpRNA', 'task_type': 'token_classification', 'label2id': {'(': 0, ')': 1, '.': 2}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA948360>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA94AAC0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-bpRNA/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-bpRNA/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-SSP-bpRNA/valid.json', 'dataset_cls': <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:38:14] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-SSP-bpRNA: task_name: RNA-SSP-bpRNA
    task_type: token_classification
    label2id: {'(': 0, ')': 1, '.': 2}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA948360>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA94AAC0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/valid.json
    dataset_cls: <class 'OmniGenome.OmniGenomeDatasetForTokenClassification'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:38:15] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "(",
        "1": ")",
        "2": "."
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "(": 0,
        ")": 1,
        ".": 2
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:38:15] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:38:15] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/train.json...
    [2025-04-19 17:38:15] [OmniGenome 0.2.4alpha4]  Loaded 9232 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/train.json
    [2025-04-19 17:38:15] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

     48%|████▊     | 4461/9232 [00:04<00:04, 1059.82it/s]

.. parsed-literal::

    [2025-04-19 17:38:19] [OmniGenome 0.2.4alpha4]  Warning: The labels <..........(((((.(.(((((((((((.((((((...)..)))))(((((((..AAAAAA.)))))))((((((.((((.(((((.(((((((..))))))))))))......)))).))))))((((((((((((.((...))...(.((.....(.((..(((((((((((((((((((........)))))).)))).aaaaaa....)))))))))..)))...)).))))))...)))))))..))))))))))))...)))))(((((((((..)))))))))..(.(((((.((.(.(((.(....).))).)))))))).)((.(((.((((....)))))())).)).(((((((((((((..))))..))))))))).................................> in the input instance do not match the label2id mapping.
    

.. parsed-literal::

     76%|███████▌  | 6973/9232 [00:06<00:02, 1109.87it/s]

.. parsed-literal::

    [2025-04-19 17:38:21] [OmniGenome 0.2.4alpha4]  Warning: The labels <((((((.(.(.((.(((((.......))))))(.A.).....(((((..()...))))).))a).))))))> in the input instance do not match the label2id mapping.
    

.. parsed-literal::

    100%|██████████| 9232/9232 [00:08<00:00, 1103.79it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:23] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 129.8156412478336, 'max_seq_len': 1024, 'min_seq_len': 12, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 9, 5,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 9,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/test.json...
    [2025-04-19 17:38:25] [OmniGenome 0.2.4alpha4]  Loaded 1161 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/test.json
    

.. parsed-literal::

    100%|██████████| 1161/1161 [00:01<00:00, 1081.56it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 130.96554694229113, 'max_seq_len': 1024, 'min_seq_len': 14, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 5,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/valid.json...
    [2025-04-19 17:38:26] [OmniGenome 0.2.4alpha4]  Loaded 1154 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-SSP-bpRNA/valid.json
    

.. parsed-literal::

    100%|██████████| 1154/1154 [00:01<00:00, 1107.81it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 128.01473136915078, 'max_seq_len': 1024, 'min_seq_len': 25, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 5,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 9,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    2,    2,  ..., -100, -100, -100])}
    [2025-04-19 17:38:27] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 289/289 [00:18<00:00, 15.94it/s]
    

.. parsed-literal::

    [2025-04-19 17:38:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.41489616879302665, 'matthews_corrcoef': 0.18007090245575402}
    [2025-04-19 17:38:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.41489616879302665, 'matthews_corrcoef': 0.18007090245575402}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.7926: 100%|██████████| 2308/2308 [06:54<00:00,  5.57it/s]
    Evaluating: 100%|██████████| 289/289 [00:18<00:00, 16.03it/s]
    

.. parsed-literal::

    [2025-04-19 17:45:59] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7506652553022602, 'matthews_corrcoef': 0.6064882861889552}
    [2025-04-19 17:45:59] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7506652553022602, 'matthews_corrcoef': 0.6064882861889552}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 291/291 [00:18<00:00, 16.04it/s]
    

.. parsed-literal::

    [2025-04-19 17:46:18] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7502921556229177, 'matthews_corrcoef': 0.6064033194534778}
    [2025-04-19 17:46:18] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7502921556229177, 'matthews_corrcoef': 0.6064033194534778}
    
    ---------------------------------------------------------- Raw Metric Records ----------------------------------------------------------
    ╒═════════════════════════╤════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                  │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M            │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M    │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-bpRNA-OmniGenome-52M       │ [0.7503] │  0.7503   │  0.7503  │   0   │   0   │ 0.7503 │ 0.7503 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M            │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M    │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-bpRNA-OmniGenome-52M       │ [0.6064] │  0.6064   │  0.6064  │   0   │   0   │ 0.6064 │ 0.6064 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M            │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M            │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ------------------------------------------- https://github.com/yangheng95/metric_visualizer -------------------------------------------
    
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-TE-Prediction.Arabidopsis Progress:  7 / 10 70.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Arabidopsis\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Arabidopsis\\__pycache__\\config.cpython-312.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis\config.py>
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-TE-Prediction.Arabidopsis from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis\config.py
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-TE-Prediction.Arabidopsis', 'task_type': 'seq_classification', 'label2id': {'0': 0, '1': 1}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DFF484A0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DFF48AE0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Arabidopsis/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Arabidopsis/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Arabidopsis/valid.json', 'dataset_cls': <class 'OmniGenome.OmniGenomeDatasetForSequenceClassification'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForSequenceClassification'>}
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:46:19] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-TE-Prediction.Arabidopsis: task_name: RNA-TE-Prediction.Arabidopsis
    task_type: seq_classification
    label2id: {'0': 0, '1': 1}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DFF484A0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DFF48AE0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/valid.json
    dataset_cls: <class 'OmniGenome.OmniGenomeDatasetForSequenceClassification'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForSequenceClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:46:23] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForSequenceClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForSequenceClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForSequenceClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "0",
        "1": "1"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "0": 0,
        "1": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:46:23] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:46:23] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/train.json...
    [2025-04-19 17:46:23] [OmniGenome 0.2.4alpha4]  Loaded 3399 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/train.json
    [2025-04-19 17:46:23] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 3399/3399 [00:08<00:00, 381.44it/s]
    

.. parsed-literal::

    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  0         		1539      		45.28%
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  1         		1860      		54.72%
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  Total samples: 3399
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 501.8608414239482, 'max_seq_len': 502, 'min_seq_len': 29, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 4, 7, 4, 4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 6, 4, 4,
            4, 7, 7, 6, 4, 5, 5, 7, 7, 7, 7, 7, 5, 4, 4, 6, 6, 7, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 6, 4, 7, 7, 4, 4, 6, 4, 4, 6, 4, 4, 6, 6, 7, 7, 6, 7,
            4, 4, 5, 4, 7, 6, 7, 5, 4, 5, 4, 4, 4, 4, 6, 7, 4, 6, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 5, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 6, 7, 7, 6, 5, 7, 6, 7,
            7, 4, 7, 7, 4, 5, 4, 7, 6, 7, 4, 4, 5, 7, 7, 7, 7, 6, 6, 7, 4, 4, 7, 4,
            4, 4, 4, 7, 7, 6, 5, 7, 7, 6, 4, 7, 6, 4, 4, 6, 5, 4, 7, 4, 5, 4, 7, 6,
            4, 7, 4, 7, 4, 5, 4, 4, 5, 4, 4, 7, 4, 4, 7, 5, 7, 5, 6, 4, 4, 6, 7, 7,
            4, 5, 4, 5, 7, 7, 7, 5, 7, 7, 6, 5, 4, 5, 4, 7, 7, 7, 5, 5, 7, 5, 7, 7,
            6, 5, 6, 7, 7, 4, 7, 7, 7, 4, 6, 5, 7, 7, 4, 7, 7, 7, 6, 5, 4, 6, 4, 5,
            7, 7, 7, 5, 7, 5, 7, 5, 5, 4, 5, 7, 5, 5, 7, 7, 7, 6, 5, 5, 4, 4, 4, 7,
            7, 7, 7, 6, 5, 7, 7, 5, 5, 5, 7, 4, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 7,
            7, 7, 7, 5, 7, 5, 7, 4, 7, 5, 7, 5, 7, 5, 7, 5, 7, 4, 7, 4, 7, 4, 7, 4,
            7, 4, 5, 4, 5, 4, 5, 4, 7, 5, 5, 4, 7, 4, 7, 6, 7, 7, 5, 4, 7, 4, 6, 4,
            4, 4, 6, 4, 4, 7, 6, 7, 4, 4, 4, 4, 7, 5, 4, 5, 6, 5, 7, 7, 7, 4, 5, 7,
            5, 6, 7, 7, 7, 7, 5, 7, 5, 7, 6, 5, 4, 7, 5, 4, 7, 5, 4, 4, 5, 4, 4, 5,
            7, 5, 5, 7, 5, 4, 4, 6, 7, 5, 4, 7, 7, 4, 5, 6, 5, 4, 5, 7, 4, 7, 7, 5,
            4, 4, 4, 4, 4, 5, 7, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 6, 4, 4, 4, 6, 4, 4,
            4, 6, 4, 7, 5, 6, 4, 4, 4, 4, 4, 6, 7, 6, 4, 4, 4, 7, 7, 7, 7, 5, 6, 4,
            7, 7, 4, 5, 6, 6, 7, 7, 7, 6, 4, 6, 7, 4, 6, 6, 4, 4, 4, 5, 6, 4, 6, 7,
            4, 4, 4, 5, 4, 5, 4, 4, 5, 4, 4, 4, 4, 6, 6, 7, 7, 5, 4, 5, 4, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(0)}
    [2025-04-19 17:46:32] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 4, 7, 7, 6, 6, 6, 7, 7, 4, 5, 4, 4, 4, 7, 4, 7, 7, 4, 7, 4, 4, 6,
            4, 5, 5, 4, 4, 4, 7, 5, 7, 6, 6, 7, 7, 6, 7, 4, 6, 7, 7, 7, 5, 4, 4, 7,
            4, 4, 5, 7, 7, 7, 7, 7, 4, 6, 7, 4, 4, 4, 7, 4, 7, 7, 6, 6, 6, 7, 4, 4,
            7, 7, 4, 4, 7, 7, 4, 4, 4, 7, 7, 7, 6, 7, 7, 7, 4, 4, 6, 4, 4, 7, 5, 7,
            5, 7, 4, 7, 7, 4, 5, 5, 4, 4, 6, 4, 5, 4, 5, 4, 7, 7, 4, 6, 5, 4, 7, 6,
            4, 7, 5, 5, 7, 7, 7, 7, 4, 6, 4, 4, 4, 7, 5, 4, 5, 4, 5, 4, 7, 4, 7, 7,
            7, 4, 4, 4, 4, 4, 4, 4, 7, 4, 7, 4, 4, 4, 7, 7, 7, 4, 4, 7, 4, 4, 7, 6,
            4, 7, 6, 6, 4, 7, 7, 6, 7, 6, 4, 4, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 4, 4,
            4, 7, 4, 6, 7, 4, 4, 6, 6, 4, 4, 4, 4, 4, 7, 7, 5, 7, 4, 7, 4, 7, 6, 7,
            4, 4, 7, 4, 4, 5, 4, 7, 6, 4, 6, 4, 7, 7, 7, 7, 6, 6, 7, 7, 7, 4, 4, 7,
            7, 5, 6, 6, 7, 7, 7, 4, 6, 4, 4, 4, 4, 4, 5, 5, 6, 7, 4, 7, 6, 7, 7, 7,
            5, 7, 7, 7, 5, 4, 4, 4, 6, 6, 4, 4, 4, 6, 4, 7, 7, 6, 4, 6, 4, 4, 5, 5,
            4, 4, 4, 6, 5, 7, 4, 6, 4, 5, 4, 4, 4, 7, 4, 7, 7, 7, 5, 5, 5, 6, 4, 5,
            6, 4, 4, 4, 6, 4, 4, 4, 5, 7, 5, 6, 7, 7, 7, 7, 5, 4, 4, 7, 5, 7, 5, 7,
            6, 4, 7, 7, 6, 5, 7, 4, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 5, 5, 4, 4, 7, 7,
            7, 6, 6, 4, 7, 7, 5, 7, 5, 4, 6, 5, 7, 7, 5, 7, 6, 6, 6, 7, 7, 7, 7, 7,
            5, 5, 4, 4, 4, 4, 7, 5, 6, 4, 4, 7, 5, 7, 7, 7, 5, 4, 5, 4, 6, 5, 7, 4,
            5, 6, 4, 7, 7, 4, 7, 7, 7, 6, 7, 4, 7, 5, 7, 6, 7, 5, 7, 5, 7, 6, 6, 7,
            4, 5, 5, 7, 5, 4, 4, 4, 6, 5, 4, 7, 5, 4, 7, 4, 4, 7, 5, 6, 6, 7, 7, 7,
            7, 5, 4, 4, 4, 4, 4, 5, 6, 7, 7, 7, 4, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 6,
            6, 7, 7, 6, 6, 6, 6, 6, 4, 7, 7, 7, 7, 6, 4, 4, 6, 4, 4, 5, 4, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:46:33] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:46:33] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/test.json...
    [2025-04-19 17:46:33] [OmniGenome 0.2.4alpha4]  Loaded 426 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/test.json
    

.. parsed-literal::

    100%|██████████| 426/426 [00:01<00:00, 386.71it/s]
    

.. parsed-literal::

    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  0         		207       		48.59%
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  1         		219       		51.41%
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Total samples: 426
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 502.0, 'max_seq_len': 502, 'min_seq_len': 502, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 6, 5, 5, 4, 6, 4, 4, 4, 7, 5, 5, 7, 7, 7, 7, 6, 4, 7, 7, 7, 6, 7,
            7, 4, 7, 5, 7, 4, 4, 7, 7, 7, 7, 4, 7, 7, 7, 7, 4, 4, 7, 7, 4, 7, 5, 5,
            4, 5, 7, 7, 4, 6, 4, 4, 5, 4, 5, 7, 7, 4, 4, 7, 7, 4, 7, 7, 6, 6, 7, 7,
            7, 7, 7, 5, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 5, 5, 7, 4, 7, 7, 5, 4, 6, 5,
            7, 4, 4, 7, 4, 5, 6, 7, 5, 7, 5, 7, 6, 5, 4, 4, 7, 5, 7, 5, 5, 7, 4, 4,
            4, 7, 4, 7, 4, 5, 4, 4, 4, 4, 4, 6, 4, 7, 7, 4, 5, 7, 6, 4, 4, 4, 7, 7,
            4, 4, 4, 4, 4, 4, 5, 7, 7, 6, 7, 6, 6, 5, 6, 5, 5, 4, 4, 7, 4, 7, 7, 7,
            7, 7, 5, 5, 5, 7, 4, 6, 4, 4, 6, 7, 7, 4, 4, 4, 4, 7, 4, 7, 7, 4, 7, 7,
            6, 4, 5, 7, 5, 5, 4, 4, 4, 4, 7, 7, 5, 4, 6, 4, 4, 5, 4, 4, 7, 7, 6, 7,
            4, 4, 4, 4, 6, 4, 4, 4, 6, 4, 4, 7, 5, 7, 7, 7, 5, 4, 4, 4, 7, 7, 5, 6,
            6, 4, 4, 7, 4, 4, 6, 7, 5, 4, 4, 7, 7, 7, 4, 7, 7, 7, 5, 5, 4, 7, 7, 7,
            7, 7, 7, 6, 7, 7, 7, 7, 5, 4, 5, 7, 5, 4, 4, 4, 4, 4, 4, 7, 7, 7, 5, 4,
            7, 5, 7, 7, 7, 7, 4, 7, 7, 7, 4, 7, 6, 7, 4, 5, 4, 4, 4, 4, 4, 5, 4, 7,
            7, 6, 4, 5, 7, 5, 5, 4, 5, 4, 7, 4, 4, 4, 4, 5, 4, 4, 7, 4, 4, 4, 4, 4,
            5, 4, 4, 7, 4, 7, 7, 7, 7, 7, 4, 6, 4, 4, 4, 4, 7, 4, 4, 6, 7, 6, 5, 7,
            6, 4, 4, 7, 6, 6, 7, 7, 6, 5, 5, 5, 4, 5, 4, 7, 6, 6, 6, 5, 4, 5, 5, 7,
            5, 5, 7, 4, 4, 7, 4, 4, 6, 7, 5, 5, 5, 5, 4, 6, 4, 4, 7, 5, 7, 4, 7, 6,
            5, 7, 7, 5, 4, 4, 4, 7, 4, 5, 4, 4, 4, 4, 5, 5, 6, 4, 4, 5, 4, 4, 4, 4,
            6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 4, 5, 4, 4, 6, 4, 4, 5, 4,
            5, 4, 4, 4, 5, 7, 7, 4, 7, 5, 7, 5, 7, 7, 7, 4, 7, 6, 7, 7, 4, 7, 5, 7,
            7, 7, 7, 6, 7, 5, 7, 6, 4, 4, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 4, 6, 5, 5, 6, 4, 6, 7, 4, 5, 7, 7, 6, 5, 7, 6, 4, 6, 5, 5, 4, 7,
            7, 7, 4, 7, 6, 7, 7, 5, 6, 6, 7, 6, 6, 4, 5, 7, 4, 4, 4, 6, 4, 7, 6, 4,
            4, 6, 4, 4, 6, 5, 7, 7, 5, 4, 5, 6, 4, 6, 7, 5, 4, 5, 6, 4, 5, 6, 5, 7,
            7, 5, 4, 4, 6, 5, 7, 7, 7, 7, 7, 5, 5, 7, 7, 6, 6, 7, 7, 5, 4, 7, 4, 7,
            6, 4, 4, 4, 7, 6, 5, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 7, 7, 4, 4, 4, 4, 5,
            6, 5, 6, 7, 5, 6, 7, 7, 7, 5, 6, 4, 4, 4, 7, 7, 6, 4, 4, 6, 4, 4, 6, 4,
            4, 4, 6, 4, 5, 6, 7, 5, 4, 5, 6, 7, 4, 4, 6, 6, 4, 4, 6, 4, 4, 7, 5, 6,
            7, 6, 7, 5, 4, 4, 5, 5, 4, 4, 4, 4, 7, 4, 4, 4, 5, 5, 6, 4, 6, 6, 7, 6,
            6, 4, 4, 5, 5, 5, 4, 5, 6, 6, 4, 6, 4, 7, 5, 5, 4, 5, 5, 4, 5, 5, 6, 4,
            4, 6, 6, 5, 4, 7, 6, 4, 4, 5, 5, 4, 5, 4, 7, 6, 6, 7, 4, 4, 5, 4, 4, 5,
            4, 4, 5, 4, 4, 5, 4, 5, 5, 6, 6, 5, 6, 5, 6, 7, 5, 4, 6, 4, 4, 4, 6, 6,
            5, 6, 5, 6, 7, 6, 4, 4, 6, 6, 7, 7, 5, 4, 4, 7, 5, 4, 7, 6, 7, 6, 6, 7,
            7, 6, 7, 6, 6, 7, 4, 6, 4, 7, 6, 4, 7, 5, 7, 5, 7, 5, 7, 7, 5, 5, 5, 5,
            5, 7, 5, 7, 4, 5, 6, 4, 7, 4, 7, 4, 4, 7, 5, 6, 6, 6, 5, 5, 7, 4, 4, 4,
            6, 6, 5, 5, 5, 4, 7, 5, 7, 5, 6, 5, 6, 7, 7, 4, 7, 4, 4, 5, 7, 4, 4, 4,
            7, 4, 6, 5, 5, 5, 4, 4, 4, 4, 7, 6, 7, 5, 6, 7, 4, 4, 4, 5, 5, 5, 7, 4,
            6, 4, 6, 6, 6, 7, 4, 4, 4, 4, 7, 4, 6, 6, 4, 4, 4, 5, 6, 7, 4, 4, 4, 4,
            4, 7, 5, 5, 7, 7, 4, 7, 5, 6, 7, 5, 7, 7, 7, 4, 7, 4, 4, 4, 5, 7, 5, 4,
            6, 7, 6, 4, 7, 4, 7, 5, 7, 7, 4, 4, 6, 4, 4, 4, 5, 5, 5, 7, 4, 6, 5, 5,
            6, 5, 4, 4, 4, 6, 4, 6, 4, 6, 4, 4, 4, 6, 6, 6, 4, 6, 6, 6, 4, 6, 6, 4,
            6, 4, 6, 7, 6, 7, 4, 6, 5, 4, 6, 4, 7, 5, 6, 6, 5, 6, 4, 4, 4, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/valid.json...
    [2025-04-19 17:46:34] [OmniGenome 0.2.4alpha4]  Loaded 424 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Arabidopsis/valid.json
    

.. parsed-literal::

    100%|██████████| 424/424 [00:01<00:00, 386.57it/s]
    

.. parsed-literal::

    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  0         		201       		47.41%
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  1         		223       		52.59%
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  Total samples: 424
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 502.0, 'max_seq_len': 502, 'min_seq_len': 502, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 5, 6, 5, 4, 6, 7, 5, 6, 4, 7, 5, 5, 4, 7, 7, 7, 6, 5, 4, 4, 5, 7,
            7, 6, 4, 4, 7, 4, 4, 7, 6, 6, 7, 7, 4, 6, 6, 7, 5, 4, 7, 6, 6, 4, 4, 7,
            4, 4, 4, 4, 4, 4, 6, 5, 7, 6, 7, 4, 7, 7, 7, 7, 4, 7, 7, 4, 5, 7, 5, 4,
            7, 6, 4, 6, 7, 4, 4, 4, 4, 7, 5, 4, 4, 7, 5, 5, 7, 5, 4, 5, 4, 7, 6, 7,
            5, 4, 7, 7, 4, 7, 4, 4, 5, 7, 5, 6, 4, 5, 5, 7, 4, 6, 7, 4, 4, 5, 4, 6,
            5, 5, 5, 4, 5, 5, 7, 7, 4, 4, 7, 6, 6, 5, 5, 5, 4, 5, 7, 4, 4, 4, 6, 5,
            5, 5, 4, 4, 4, 7, 7, 7, 6, 4, 4, 4, 7, 4, 4, 5, 6, 6, 7, 5, 4, 4, 4, 7,
            5, 7, 7, 4, 4, 5, 4, 7, 7, 7, 4, 4, 6, 7, 7, 4, 4, 5, 4, 4, 4, 7, 6, 4,
            7, 4, 4, 4, 7, 7, 4, 6, 7, 4, 6, 7, 4, 7, 4, 7, 6, 4, 7, 7, 4, 4, 4, 5,
            7, 5, 4, 4, 4, 7, 5, 7, 4, 4, 5, 4, 4, 6, 6, 4, 7, 6, 5, 4, 4, 4, 7, 6,
            4, 7, 5, 4, 7, 5, 6, 4, 7, 5, 4, 7, 5, 4, 5, 4, 7, 6, 7, 5, 4, 4, 7, 7,
            4, 4, 4, 4, 7, 7, 7, 4, 7, 7, 7, 4, 4, 5, 6, 7, 7, 6, 7, 6, 7, 4, 4, 5,
            6, 5, 4, 7, 4, 7, 4, 4, 4, 4, 7, 6, 4, 7, 4, 7, 4, 4, 7, 7, 4, 7, 7, 4,
            4, 7, 7, 7, 5, 4, 5, 7, 4, 4, 4, 4, 7, 7, 7, 7, 6, 4, 4, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 4, 7, 6, 4, 4, 7, 7, 6, 7, 5, 6, 6, 6, 4, 4, 6, 5, 6, 6, 4,
            4, 4, 4, 6, 7, 6, 7, 7, 4, 5, 5, 6, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 4, 7,
            5, 4, 4, 5, 6, 4, 7, 6, 4, 5, 6, 7, 4, 5, 5, 4, 4, 5, 4, 6, 7, 6, 4, 4,
            7, 6, 4, 7, 7, 7, 6, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 5, 5, 5, 4, 5, 4,
            5, 4, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7,
            7, 7, 6, 6, 6, 4, 7, 7, 5, 6, 4, 6, 4, 5, 6, 4, 4, 4, 5, 5, 5, 7, 4, 6,
            5, 7, 5, 7, 6, 4, 7, 7, 7, 7, 7, 7, 7, 5, 4, 4, 7, 7, 5, 5, 6, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(0)}
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6, 4, 7, 7, 4, 7, 7, 7, 4, 4, 7, 7, 4, 4, 4, 4, 6, 4, 7, 4, 4, 7,
            7, 4, 5, 4, 7, 4, 4, 6, 4, 7, 4, 7, 7, 7, 6, 6, 4, 7, 4, 4, 7, 7, 7, 5,
            6, 4, 4, 4, 7, 4, 7, 5, 4, 7, 4, 7, 7, 7, 7, 7, 4, 6, 6, 7, 7, 6, 5, 5,
            4, 4, 4, 4, 7, 4, 4, 6, 4, 5, 5, 7, 4, 5, 7, 5, 7, 7, 7, 4, 7, 4, 7, 7,
            4, 6, 4, 7, 7, 4, 6, 6, 7, 5, 7, 4, 5, 5, 4, 7, 5, 4, 4, 4, 7, 7, 5, 7,
            7, 7, 7, 7, 6, 5, 4, 4, 4, 7, 4, 6, 4, 4, 4, 5, 4, 6, 6, 4, 4, 4, 4, 7,
            7, 5, 7, 4, 4, 5, 4, 4, 5, 7, 7, 4, 4, 7, 4, 7, 7, 7, 7, 5, 7, 7, 7, 4,
            4, 4, 4, 4, 5, 7, 7, 4, 7, 5, 7, 5, 4, 4, 6, 5, 5, 4, 4, 4, 4, 7, 5, 5,
            4, 4, 5, 6, 4, 7, 4, 7, 5, 7, 4, 7, 5, 6, 4, 7, 4, 7, 6, 6, 7, 4, 7, 7,
            4, 7, 5, 4, 4, 7, 5, 4, 4, 4, 4, 6, 4, 7, 7, 4, 4, 7, 4, 7, 7, 6, 5, 4,
            7, 4, 4, 4, 7, 6, 7, 7, 7, 7, 6, 4, 7, 4, 4, 7, 7, 7, 7, 6, 7, 7, 4, 4,
            5, 4, 4, 7, 4, 6, 4, 4, 7, 4, 4, 6, 7, 6, 7, 6, 4, 4, 4, 7, 7, 4, 7, 6,
            4, 4, 4, 7, 7, 4, 4, 5, 7, 6, 6, 4, 7, 7, 5, 7, 4, 4, 7, 7, 5, 7, 4, 7,
            5, 5, 5, 7, 7, 6, 7, 4, 4, 5, 7, 7, 6, 6, 6, 4, 4, 6, 6, 6, 7, 6, 7, 7,
            4, 4, 6, 7, 6, 7, 7, 4, 4, 5, 4, 5, 4, 5, 7, 5, 6, 6, 7, 4, 4, 4, 7, 5,
            4, 5, 7, 5, 4, 7, 6, 6, 4, 5, 7, 7, 7, 6, 4, 4, 7, 7, 7, 7, 4, 7, 4, 7,
            7, 7, 6, 6, 6, 5, 5, 5, 4, 7, 4, 4, 7, 6, 6, 4, 7, 5, 4, 4, 7, 5, 6, 7,
            4, 4, 4, 5, 5, 5, 4, 7, 5, 4, 4, 4, 6, 4, 4, 6, 5, 7, 5, 4, 4, 4, 4, 5,
            4, 7, 5, 7, 4, 4, 7, 7, 7, 7, 4, 7, 7, 5, 5, 4, 7, 7, 7, 7, 7, 6, 6, 5,
            4, 5, 5, 7, 5, 4, 4, 4, 4, 7, 4, 7, 4, 7, 4, 4, 4, 5, 4, 7, 7, 7, 4, 7,
            4, 7, 7, 4, 7, 7, 6, 5, 6, 4, 4, 4, 7, 7, 7, 5, 5, 6, 4, 7, 5, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:46:35] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 106/106 [00:06<00:00, 16.22it/s]
    

.. parsed-literal::

    [2025-04-19 17:46:42] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3216, 'matthews_corrcoef': 0.0}
    [2025-04-19 17:46:42] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3216, 'matthews_corrcoef': 0.0}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.6482: 100%|██████████| 850/850 [02:25<00:00,  5.85it/s]
    Evaluating: 100%|██████████| 106/106 [00:06<00:00, 16.55it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:14] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.6495471014492753, 'matthews_corrcoef': 0.30772444027319046}
    [2025-04-19 17:49:14] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.6495471014492753, 'matthews_corrcoef': 0.30772444027319046}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 107/107 [00:06<00:00, 16.52it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.6601828557403203, 'matthews_corrcoef': 0.3295066167896064}
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.6601828557403203, 'matthews_corrcoef': 0.3295066167896064}
    
    --------------------------------------------------------------- Raw Metric Records ---------------------------------------------------------------
    ╒═════════════════════════╤══════════════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                            │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪══════════════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M                      │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M              │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M           │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                 │ [0.7503] │  0.7503   │  0.7503  │   0   │   0   │ 0.7503 │ 0.7503 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M │ [0.6602] │  0.6602   │  0.6602  │   0   │   0   │ 0.6602 │ 0.6602 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M                      │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M              │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M           │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                 │ [0.6064] │  0.6064   │  0.6064  │   0   │   0   │ 0.6064 │ 0.6064 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M │ [0.3295] │  0.3295   │  0.3295  │   0   │   0   │ 0.3295 │ 0.3295 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M                      │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M                      │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧══════════════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ------------------------------------------------ https://github.com/yangheng95/metric_visualizer ------------------------------------------------
    
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-TE-Prediction.Rice Progress:  8 / 10 80.0%
    FindFile Warning --> multiple targets ['__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Rice\\config.py', '__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Rice\\__pycache__\\config.cpython-312.pyc'] found, only return the shortest path: <__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice\config.py>
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-TE-Prediction.Rice from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice\config.py
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-TE-Prediction.Rice', 'task_type': 'seq_classification', 'label2id': {'0': 0, '1': 1}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150EA0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA847CE0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Rice/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Rice/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-TE-Prediction\\Rice/valid.json', 'dataset_cls': <class 'OmniGenome.OmniGenomeDatasetForSequenceClassification'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForSequenceClassification'>}
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:49:22] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-TE-Prediction.Rice: task_name: RNA-TE-Prediction.Rice
    task_type: seq_classification
    label2id: {'0': 0, '1': 1}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA150EA0>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DA847CE0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/valid.json
    dataset_cls: <class 'OmniGenome.OmniGenomeDatasetForSequenceClassification'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForSequenceClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:49:23] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForSequenceClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForSequenceClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForSequenceClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "0",
        "1": "1"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "0": 0,
        "1": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:49:23] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:49:23] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/train.json...
    [2025-04-19 17:49:23] [OmniGenome 0.2.4alpha4]  Loaded 4697 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/train.json
    [2025-04-19 17:49:23] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    100%|██████████| 4697/4697 [00:12<00:00, 388.44it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  0         		2195      		46.73%
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  1         		2502      		53.27%
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:35] [OmniGenome 0.2.4alpha4]  Total samples: 4697
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 502.0, 'max_seq_len': 502, 'min_seq_len': 502, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 4, 4, 5, 7, 4, 4, 4, 4, 4, 6, 7, 6, 6, 4, 6, 7, 4, 6, 7, 7, 5, 5,
            7, 7, 6, 4, 5, 6, 6, 4, 4, 6, 4, 4, 6, 6, 6, 4, 6, 5, 4, 4, 4, 4, 7, 4,
            6, 4, 4, 6, 4, 7, 4, 7, 7, 5, 7, 5, 4, 6, 7, 7, 6, 4, 7, 5, 7, 6, 5, 4,
            6, 7, 7, 6, 7, 7, 6, 7, 7, 4, 6, 6, 7, 5, 4, 5, 7, 4, 7, 4, 7, 7, 5, 4,
            6, 4, 4, 4, 7, 5, 6, 5, 4, 6, 7, 7, 6, 5, 7, 6, 7, 7, 6, 7, 7, 7, 4, 4,
            4, 7, 7, 6, 7, 6, 7, 6, 7, 6, 4, 5, 4, 6, 5, 4, 6, 4, 5, 4, 6, 5, 7, 4,
            4, 7, 7, 4, 7, 5, 4, 6, 7, 4, 5, 4, 5, 6, 7, 4, 7, 4, 7, 6, 4, 6, 5, 4,
            4, 7, 4, 5, 7, 4, 6, 7, 6, 4, 4, 7, 5, 7, 6, 7, 4, 5, 7, 4, 4, 7, 7, 7,
            4, 4, 5, 6, 4, 6, 4, 6, 7, 4, 7, 7, 7, 7, 5, 7, 4, 7, 4, 7, 4, 5, 4, 4,
            4, 7, 4, 5, 4, 4, 5, 4, 6, 5, 7, 4, 4, 4, 5, 7, 6, 7, 6, 5, 5, 4, 5, 7,
            6, 6, 5, 6, 5, 5, 6, 4, 4, 7, 4, 5, 6, 7, 4, 5, 6, 6, 4, 5, 4, 6, 4, 6,
            5, 7, 5, 4, 6, 6, 5, 4, 4, 7, 5, 4, 6, 6, 6, 6, 4, 6, 5, 4, 6, 5, 4, 4,
            4, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 6, 7, 7, 6, 6, 7, 6, 5, 5, 4, 4, 6, 5,
            4, 5, 4, 4, 5, 7, 4, 4, 4, 5, 5, 5, 4, 4, 5, 7, 6, 5, 4, 5, 5, 5, 4, 4,
            4, 4, 4, 5, 7, 4, 4, 7, 5, 4, 6, 5, 4, 7, 7, 7, 5, 4, 6, 7, 7, 5, 6, 5,
            7, 7, 7, 4, 6, 7, 7, 4, 6, 7, 4, 5, 7, 4, 5, 5, 4, 5, 5, 7, 6, 5, 4, 7,
            5, 7, 5, 7, 7, 7, 4, 5, 5, 4, 4, 5, 4, 5, 7, 4, 7, 4, 7, 4, 4, 5, 5, 5,
            6, 5, 4, 6, 7, 6, 6, 4, 5, 5, 7, 6, 5, 4, 6, 7, 5, 4, 7, 5, 7, 5, 4, 5,
            7, 4, 4, 7, 7, 5, 4, 6, 7, 6, 4, 4, 6, 5, 5, 4, 5, 5, 4, 6, 7, 4, 5, 7,
            4, 6, 7, 4, 5, 6, 6, 5, 7, 5, 7, 4, 4, 7, 5, 4, 6, 7, 7, 5, 6, 5, 6, 7,
            7, 7, 6, 5, 7, 4, 4, 7, 7, 4, 4, 5, 7, 5, 7, 6, 5, 5, 4, 7, 5, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 4, 6, 4, 7, 7, 7, 5, 6, 7, 4, 4, 4, 5, 4, 4, 5, 4, 7, 4, 6, 4, 6,
            6, 4, 4, 6, 4, 4, 7, 6, 4, 4, 6, 6, 6, 4, 4, 4, 4, 7, 5, 5, 7, 4, 5, 5,
            6, 4, 6, 6, 5, 4, 4, 7, 7, 6, 7, 4, 7, 7, 7, 6, 7, 7, 7, 7, 6, 6, 7, 4,
            6, 4, 7, 4, 7, 6, 5, 7, 5, 7, 7, 5, 4, 4, 6, 5, 4, 5, 7, 7, 6, 4, 4, 5,
            5, 5, 4, 5, 7, 7, 6, 6, 4, 7, 7, 4, 5, 6, 6, 5, 6, 4, 6, 6, 5, 4, 6, 4,
            7, 4, 6, 4, 4, 6, 5, 4, 6, 6, 4, 5, 6, 4, 4, 6, 4, 6, 5, 4, 4, 7, 6, 4,
            5, 4, 5, 6, 4, 7, 4, 7, 6, 5, 4, 5, 6, 7, 5, 6, 7, 6, 6, 7, 6, 6, 4, 4,
            4, 4, 4, 7, 5, 7, 6, 6, 6, 7, 4, 5, 6, 7, 4, 7, 4, 7, 7, 7, 5, 5, 5, 6,
            4, 5, 4, 4, 4, 5, 5, 6, 6, 7, 7, 4, 5, 4, 4, 7, 4, 4, 6, 4, 5, 5, 6, 4,
            5, 5, 6, 4, 4, 4, 5, 4, 5, 6, 7, 4, 7, 6, 6, 6, 7, 7, 5, 6, 6, 6, 4, 4,
            4, 6, 6, 6, 4, 7, 5, 5, 5, 5, 5, 6, 4, 4, 7, 4, 7, 7, 6, 6, 6, 7, 4, 6,
            5, 5, 6, 7, 7, 6, 7, 7, 4, 4, 4, 5, 5, 4, 6, 6, 7, 5, 6, 4, 4, 7, 4, 5,
            7, 7, 7, 4, 7, 6, 4, 4, 4, 7, 6, 6, 6, 5, 6, 6, 4, 6, 7, 4, 7, 5, 5, 6,
            4, 4, 4, 5, 7, 6, 7, 4, 6, 5, 7, 4, 6, 4, 6, 7, 4, 6, 5, 7, 4, 7, 7, 7,
            5, 5, 4, 7, 4, 6, 5, 7, 6, 5, 5, 4, 6, 5, 4, 4, 4, 4, 7, 6, 5, 5, 5, 4,
            7, 4, 5, 6, 4, 4, 6, 7, 5, 4, 4, 7, 7, 7, 5, 7, 7, 5, 6, 6, 7, 7, 4, 6,
            4, 4, 4, 7, 4, 7, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 6, 7, 4,
            6, 7, 4, 7, 7, 6, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 5, 5, 4, 6, 6, 7, 7,
            5, 7, 7, 5, 7, 7, 7, 5, 7, 6, 6, 4, 4, 4, 6, 4, 5, 4, 4, 7, 4, 7, 7, 7,
            5, 7, 7, 7, 5, 4, 7, 5, 5, 7, 7, 7, 7, 6, 5, 4, 7, 7, 6, 4, 4, 4, 7, 4,
            4, 5, 4, 4, 4, 7, 7, 6, 4, 4, 4, 7, 5, 4, 4, 4, 4, 7, 4, 4, 7, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/test.json...
    [2025-04-19 17:49:36] [OmniGenome 0.2.4alpha4]  Loaded 588 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/test.json
    

.. parsed-literal::

    100%|██████████| 588/588 [00:01<00:00, 372.04it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  0         		258       		43.88%
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  1         		330       		56.12%
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Total samples: 588
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 502.0, 'max_seq_len': 502, 'min_seq_len': 502, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 4, 6, 6, 6, 5, 7, 4, 7, 7, 4, 5, 5, 7, 4, 7, 5, 4, 4, 6, 6, 6, 6,
            5, 7, 7, 6, 4, 4, 5, 5, 4, 6, 7, 4, 7, 4, 4, 7, 7, 5, 7, 7, 6, 7, 5, 7,
            7, 7, 7, 6, 6, 7, 7, 6, 5, 7, 7, 6, 4, 7, 6, 7, 5, 6, 7, 4, 5, 7, 4, 5,
            6, 7, 4, 6, 4, 7, 5, 5, 7, 7, 6, 7, 4, 5, 5, 4, 4, 5, 6, 7, 4, 5, 5, 5,
            5, 4, 4, 7, 4, 5, 5, 5, 7, 5, 7, 4, 7, 4, 7, 5, 5, 6, 6, 7, 5, 7, 4, 5,
            6, 6, 6, 7, 4, 7, 5, 4, 5, 5, 5, 6, 7, 5, 6, 4, 7, 4, 5, 7, 5, 5, 7, 4,
            4, 7, 4, 4, 7, 5, 7, 4, 6, 4, 7, 7, 4, 7, 4, 4, 7, 4, 4, 7, 5, 5, 7, 4,
            7, 6, 5, 7, 6, 4, 4, 5, 5, 4, 4, 4, 5, 4, 6, 6, 6, 5, 5, 7, 4, 4, 4, 4,
            6, 4, 4, 7, 5, 7, 4, 7, 7, 6, 6, 7, 4, 4, 4, 7, 7, 7, 7, 7, 7, 4, 7, 4,
            7, 4, 7, 4, 7, 6, 7, 7, 7, 6, 7, 4, 6, 7, 6, 6, 5, 7, 5, 4, 4, 4, 4, 6,
            5, 7, 4, 4, 7, 4, 4, 7, 4, 4, 4, 4, 4, 4, 7, 4, 7, 4, 5, 6, 7, 7, 4, 4,
            4, 4, 4, 7, 4, 7, 4, 7, 7, 7, 4, 4, 4, 7, 4, 4, 5, 7, 7, 7, 4, 4, 4, 4,
            7, 5, 4, 4, 6, 7, 7, 5, 4, 4, 4, 4, 4, 6, 7, 7, 7, 4, 4, 4, 7, 7, 7, 7,
            6, 6, 7, 7, 4, 7, 7, 4, 7, 7, 7, 6, 4, 5, 7, 7, 4, 7, 7, 4, 6, 4, 7, 5,
            4, 4, 5, 5, 6, 4, 6, 6, 4, 6, 6, 5, 7, 4, 7, 7, 7, 4, 5, 5, 4, 5, 7, 5,
            5, 5, 7, 6, 5, 5, 4, 5, 7, 6, 5, 4, 6, 5, 4, 5, 4, 5, 5, 6, 4, 5, 4, 5,
            6, 7, 6, 4, 5, 4, 5, 6, 7, 4, 5, 4, 5, 5, 5, 7, 5, 5, 5, 6, 7, 6, 4, 5,
            6, 5, 5, 6, 5, 5, 6, 5, 6, 5, 6, 7, 6, 7, 6, 5, 7, 4, 5, 6, 6, 5, 5, 4,
            5, 4, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 7, 4, 5, 4, 5, 6, 5, 6, 4, 6, 4, 6,
            4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6,
            4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 4, 6, 5, 6, 6, 5, 6, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 5, 7, 4, 4, 4, 4, 7, 7, 7, 5, 5, 6, 4, 4, 5, 5, 5, 7, 5, 5, 5, 5,
            4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
            4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 6, 6, 5, 5, 4,
            4, 7, 7, 7, 7, 7, 7, 7, 6, 4, 7, 7, 7, 7, 7, 7, 4, 7, 4, 4, 7, 7, 7, 7,
            7, 7, 7, 7, 6, 7, 6, 4, 4, 7, 7, 7, 7, 6, 6, 5, 4, 6, 7, 4, 7, 7, 7, 6,
            7, 7, 5, 4, 4, 4, 4, 7, 7, 5, 5, 6, 4, 4, 4, 7, 7, 7, 7, 6, 4, 4, 4, 4,
            7, 7, 7, 5, 4, 7, 5, 4, 7, 4, 7, 5, 4, 6, 5, 6, 5, 5, 5, 7, 7, 4, 6, 4,
            7, 4, 5, 4, 4, 4, 7, 4, 5, 6, 7, 7, 4, 4, 6, 6, 4, 4, 4, 7, 4, 7, 7, 6,
            4, 4, 5, 5, 5, 7, 6, 5, 5, 7, 6, 7, 7, 7, 6, 7, 6, 4, 5, 7, 7, 7, 7, 5,
            4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 7, 7, 4, 7, 6, 7, 7, 7, 6, 6, 5,
            7, 5, 5, 5, 5, 5, 6, 7, 4, 5, 7, 5, 7, 7, 7, 7, 7, 6, 5, 4, 4, 5, 4, 7,
            6, 7, 4, 4, 4, 7, 4, 5, 4, 7, 4, 6, 4, 6, 7, 5, 5, 7, 6, 6, 7, 5, 4, 4,
            4, 4, 4, 7, 6, 5, 4, 7, 6, 7, 4, 4, 7, 6, 6, 4, 4, 7, 4, 4, 5, 4, 7, 7,
            7, 6, 4, 6, 7, 7, 7, 7, 4, 4, 4, 5, 6, 7, 4, 6, 7, 5, 4, 7, 4, 6, 6, 4,
            7, 4, 4, 7, 4, 4, 7, 7, 7, 6, 5, 4, 5, 6, 5, 5, 5, 4, 4, 4, 7, 5, 4, 7,
            4, 6, 5, 5, 7, 7, 5, 5, 6, 7, 7, 7, 7, 7, 6, 4, 7, 5, 6, 6, 4, 5, 6, 6,
            5, 7, 5, 6, 6, 4, 7, 4, 4, 4, 5, 5, 6, 4, 4, 4, 5, 6, 5, 5, 7, 4, 6, 4,
            7, 6, 4, 7, 6, 4, 4, 4, 6, 5, 5, 7, 5, 7, 6, 5, 6, 4, 7, 7, 5, 6, 4, 7,
            6, 5, 6, 5, 6, 7, 4, 4, 5, 5, 5, 5, 5, 7, 5, 7, 5, 6, 4, 4, 6, 5, 6, 4,
            7, 4, 4, 6, 5, 5, 5, 5, 6, 7, 6, 6, 6, 4, 6, 4, 6, 4, 5, 6, 4, 6, 5, 6,
            5, 6, 5, 6, 5, 7, 5, 5, 6, 5, 7, 5, 5, 6, 4, 6, 5, 4, 6, 4, 4, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/valid.json...
    [2025-04-19 17:49:38] [OmniGenome 0.2.4alpha4]  Loaded 587 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-TE-Prediction\Rice/valid.json
    

.. parsed-literal::

    100%|██████████| 587/587 [00:01<00:00, 381.31it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  
    Label Distribution:
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  Label     		Count     		Percentage
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  0         		259       		44.12%
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  1         		328       		55.88%
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  ----------------------------------------
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  Total samples: 587
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=504, label_padding_length=0
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 502.0, 'max_seq_len': 502, 'min_seq_len': 502, 'avg_label_len': 1.0, 'max_label_len': 1, 'min_label_len': 1}
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 6, 7, 6, 5, 5, 4, 6, 5, 4, 4, 7, 6, 5, 4, 7, 4, 5, 4, 7, 7, 4, 7,
            4, 4, 4, 7, 4, 4, 5, 4, 5, 7, 7, 4, 7, 5, 6, 6, 4, 7, 4, 4, 6, 6, 6, 7,
            7, 4, 5, 6, 4, 6, 6, 4, 7, 5, 4, 4, 7, 4, 4, 5, 5, 6, 4, 7, 6, 4, 5, 5,
            5, 5, 6, 7, 4, 7, 7, 4, 5, 4, 4, 7, 6, 4, 4, 5, 4, 4, 7, 4, 6, 6, 6, 7,
            4, 6, 7, 6, 6, 6, 6, 6, 4, 5, 5, 5, 6, 4, 4, 4, 5, 5, 5, 6, 7, 7, 4, 5,
            5, 5, 6, 4, 5, 6, 4, 4, 7, 5, 7, 6, 5, 4, 7, 6, 4, 6, 6, 5, 6, 7, 5, 4,
            6, 6, 6, 6, 6, 5, 5, 5, 4, 5, 6, 6, 6, 6, 7, 5, 6, 6, 5, 5, 5, 6, 5, 5,
            5, 5, 7, 6, 7, 5, 6, 5, 6, 6, 6, 5, 7, 6, 4, 5, 5, 5, 6, 6, 5, 4, 7, 6,
            6, 5, 5, 5, 7, 5, 4, 7, 6, 6, 5, 7, 6, 5, 4, 5, 5, 5, 7, 4, 6, 5, 7, 4,
            5, 5, 5, 5, 5, 4, 5, 4, 4, 5, 5, 4, 5, 5, 5, 7, 5, 5, 5, 4, 4, 6, 5, 6,
            7, 6, 7, 6, 6, 5, 5, 7, 6, 6, 5, 4, 6, 7, 6, 6, 5, 4, 5, 6, 7, 4, 7, 6,
            6, 4, 7, 6, 7, 4, 5, 6, 7, 4, 5, 4, 7, 6, 5, 4, 5, 7, 5, 6, 7, 6, 7, 6,
            5, 7, 5, 4, 5, 4, 7, 5, 5, 4, 5, 4, 7, 6, 7, 6, 7, 6, 7, 6, 5, 7, 7, 5,
            7, 7, 5, 4, 5, 4, 7, 4, 6, 5, 7, 4, 6, 6, 6, 7, 7, 7, 6, 7, 7, 4, 6, 5,
            7, 4, 6, 6, 7, 5, 7, 4, 7, 4, 4, 4, 4, 5, 4, 7, 5, 6, 5, 5, 4, 7, 7, 6,
            5, 7, 6, 5, 7, 5, 4, 6, 5, 7, 4, 6, 5, 4, 6, 5, 5, 4, 5, 6, 4, 4, 6, 4,
            5, 4, 4, 4, 6, 7, 5, 4, 6, 4, 6, 6, 7, 5, 4, 5, 4, 5, 4, 6, 5, 7, 5, 4,
            7, 5, 6, 7, 5, 7, 5, 6, 4, 6, 5, 7, 4, 5, 5, 7, 4, 6, 5, 7, 6, 6, 4, 4,
            6, 4, 4, 6, 4, 4, 6, 4, 6, 6, 4, 4, 6, 4, 4, 6, 4, 6, 6, 4, 6, 6, 4, 6,
            6, 4, 6, 5, 7, 5, 6, 7, 5, 6, 6, 4, 6, 7, 7, 7, 4, 6, 7, 7, 6, 6, 7, 6,
            4, 6, 7, 7, 5, 6, 5, 5, 6, 6, 4, 6, 4, 6, 4, 7, 5, 6, 4, 7, 5, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 4, 5, 7, 6, 4, 7, 5, 7, 5, 5, 4, 6, 5, 7, 6, 7, 7, 5, 6, 5, 4, 7,
            6, 6, 7, 7, 6, 6, 7, 6, 5, 5, 5, 5, 7, 5, 7, 5, 5, 7, 7, 5, 5, 5, 6, 6,
            6, 7, 4, 6, 7, 7, 6, 6, 6, 6, 4, 4, 7, 6, 4, 4, 4, 7, 6, 4, 5, 6, 5, 7,
            6, 6, 6, 4, 7, 7, 6, 5, 4, 5, 6, 6, 7, 7, 4, 6, 7, 4, 4, 4, 6, 7, 6, 7,
            7, 7, 4, 4, 4, 5, 7, 6, 6, 4, 6, 7, 7, 5, 7, 7, 4, 5, 7, 6, 5, 7, 7, 6,
            5, 7, 7, 6, 6, 7, 7, 7, 4, 4, 5, 5, 4, 5, 4, 4, 5, 5, 5, 6, 6, 7, 5, 5,
            5, 7, 5, 7, 6, 6, 7, 6, 7, 6, 5, 6, 6, 6, 5, 7, 6, 6, 6, 5, 7, 4, 6, 5,
            7, 6, 6, 7, 6, 6, 5, 5, 6, 6, 5, 7, 5, 4, 6, 7, 6, 6, 5, 5, 5, 7, 5, 6,
            5, 7, 7, 7, 4, 7, 7, 7, 4, 5, 7, 5, 5, 7, 7, 4, 4, 6, 4, 4, 7, 5, 6, 7,
            7, 5, 6, 5, 7, 7, 5, 4, 6, 5, 7, 6, 4, 5, 5, 7, 5, 7, 5, 7, 5, 5, 6, 5,
            5, 5, 5, 6, 5, 5, 7, 5, 5, 5, 5, 7, 7, 5, 5, 6, 7, 5, 7, 5, 7, 5, 5, 6,
            5, 6, 5, 7, 5, 5, 4, 6, 5, 7, 5, 5, 4, 5, 6, 5, 5, 4, 4, 7, 7, 4, 4, 4,
            5, 5, 5, 7, 4, 5, 7, 5, 5, 7, 5, 7, 7, 6, 5, 6, 7, 7, 6, 5, 6, 7, 7, 4,
            7, 5, 7, 6, 5, 6, 6, 7, 7, 5, 5, 6, 7, 5, 7, 5, 5, 5, 4, 5, 7, 4, 5, 7,
            5, 6, 5, 6, 5, 4, 5, 4, 5, 6, 7, 4, 6, 4, 4, 4, 5, 6, 6, 4, 4, 6, 6, 4,
            6, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 6, 6, 5, 6, 5, 5, 6, 4, 6, 7, 4,
            4, 5, 6, 5, 4, 6, 4, 7, 4, 6, 4, 6, 4, 6, 7, 7, 6, 7, 7, 7, 5, 5, 6, 5,
            4, 5, 5, 6, 4, 6, 5, 6, 5, 6, 7, 5, 5, 6, 5, 6, 6, 5, 4, 6, 7, 6, 7, 6,
            7, 6, 7, 6, 7, 6, 7, 4, 6, 5, 6, 6, 7, 4, 6, 5, 6, 5, 4, 5, 6, 5, 6, 5,
            6, 6, 5, 7, 5, 6, 5, 6, 6, 6, 5, 6, 5, 6, 5, 5, 6, 6, 7, 7, 6, 5, 4, 5,
            5, 4, 6, 5, 7, 6, 7, 6, 5, 6, 7, 7, 6, 5, 6, 6, 5, 6, 4, 5, 6, 2, 1, 1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), 'labels': tensor(1)}
    [2025-04-19 17:49:39] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 147/147 [00:09<00:00, 15.78it/s]
    

.. parsed-literal::

    [2025-04-19 17:49:49] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3061465721040189, 'matthews_corrcoef': 0.0}
    [2025-04-19 17:49:49] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3061465721040189, 'matthews_corrcoef': 0.0}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.6317: 100%|██████████| 1175/1175 [03:21<00:00,  5.84it/s]
    Evaluating: 100%|██████████| 147/147 [00:08<00:00, 16.49it/s]
    

.. parsed-literal::

    [2025-04-19 17:53:20] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7186528255126137, 'matthews_corrcoef': 0.4389950704998246}
    [2025-04-19 17:53:20] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7186528255126137, 'matthews_corrcoef': 0.4389950704998246}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 147/147 [00:08<00:00, 16.40it/s]
    

.. parsed-literal::

    [2025-04-19 17:53:29] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7427322170034104, 'matthews_corrcoef': 0.49504530737460883}
    [2025-04-19 17:53:29] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.7427322170034104, 'matthews_corrcoef': 0.49504530737460883}
    
    --------------------------------------------------------------- Raw Metric Records ---------------------------------------------------------------
    ╒═════════════════════════╤══════════════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                            │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪══════════════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M                      │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M              │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M           │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                 │ [0.7503] │  0.7503   │  0.7503  │   0   │   0   │ 0.7503 │ 0.7503 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M │ [0.6602] │  0.6602   │  0.6602  │   0   │   0   │ 0.6602 │ 0.6602 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M        │ [0.7427] │  0.7427   │  0.7427  │   0   │   0   │ 0.7427 │ 0.7427 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M                      │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M              │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M           │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                 │ [0.6064] │  0.6064   │  0.6064  │   0   │   0   │ 0.6064 │ 0.6064 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M │ [0.3295] │  0.3295   │  0.3295  │   0   │   0   │ 0.3295 │ 0.3295 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M        │ [0.495]  │   0.495   │  0.495   │   0   │   0   │ 0.495  │ 0.495  │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M                      │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼──────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M                      │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧══════════════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ------------------------------------------------ https://github.com/yangheng95/metric_visualizer ------------------------------------------------
    
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-Region-Classification.Arabidopsis Progress:  9 / 10 90.0%
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-Region-Classification.Arabidopsis from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis\config.py
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-Region-Classification.Arabidopsis', 'task_type': 'token_classification', 'label2id': {'3utr': 0, 'cds': 1, '5utr': 2}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC47100>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC479C0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Arabidopsis/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Arabidopsis/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Arabidopsis/valid.json', 'dataset_cls': <class 'config.RegionClassificationDataset'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 17:53:30] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-Region-Classification.Arabidopsis: task_name: RNA-Region-Classification.Arabidopsis
    task_type: token_classification
    label2id: {'3utr': 0, 'cds': 1, '5utr': 2}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC47100>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC479C0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/valid.json
    dataset_cls: <class 'config.RegionClassificationDataset'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 17:53:35] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "3utr",
        "1": "cds",
        "2": "5utr"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "3utr": 0,
        "5utr": 2,
        "cds": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 17:53:35] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:53:35] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/train.json...
    [2025-04-19 17:53:35] [OmniGenome 0.2.4alpha4]  Loaded 17860 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/train.json
    [2025-04-19 17:53:35] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_dataset.py:353: UserWarning: The 'sequence' field is missing in the raw dataset.
      warnings.warn("The 'sequence' field is missing in the raw dataset.")
    100%|██████████| 17860/17860 [02:23<00:00, 124.63it/s]
    

.. parsed-literal::

    [2025-04-19 17:55:59] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 968.9286674132139, 'max_seq_len': 1024, 'min_seq_len': 241, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7,  ..., 7, 4, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7,  ..., 4, 4, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/test.json...
    [2025-04-19 17:56:00] [OmniGenome 0.2.4alpha4]  Loaded 2232 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/test.json
    

.. parsed-literal::

    100%|██████████| 2232/2232 [00:17<00:00, 126.01it/s]
    

.. parsed-literal::

    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 963.9560931899641, 'max_seq_len': 1024, 'min_seq_len': 220, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 7,  ..., 7, 6, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 5, 7,  ..., 5, 5, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    2,    2, -100])}
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/valid.json...
    [2025-04-19 17:56:18] [OmniGenome 0.2.4alpha4]  Loaded 2233 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Arabidopsis/valid.json
    

.. parsed-literal::

    100%|██████████| 2233/2233 [00:17<00:00, 125.20it/s]
    

.. parsed-literal::

    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 968.8920734437976, 'max_seq_len': 1024, 'min_seq_len': 307, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 4,  ..., 6, 4, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 4,  ..., 1, 1, 1]), 'attention_mask': tensor([1, 1, 1,  ..., 0, 0, 0]), 'labels': tensor([-100,    0,    0,  ..., -100, -100, -100])}
    [2025-04-19 17:56:36] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 559/559 [00:34<00:00, 16.02it/s]
    

.. parsed-literal::

    [2025-04-19 17:57:13] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3125039822115352, 'matthews_corrcoef': 0.0028170192809846384}
    [2025-04-19 17:57:13] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.3125039822115352, 'matthews_corrcoef': 0.0028170192809846384}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.5833: 100%|██████████| 4465/4465 [13:19<00:00,  5.58it/s]
    Evaluating: 100%|██████████| 559/559 [00:34<00:00, 16.20it/s]
    

.. parsed-literal::

    [2025-04-19 18:11:10] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9573647225361115, 'matthews_corrcoef': 0.9478401072462141}
    [2025-04-19 18:11:10] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9573647225361115, 'matthews_corrcoef': 0.9478401072462141}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 558/558 [00:34<00:00, 16.14it/s]
    

.. parsed-literal::

    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9601409160538914, 'matthews_corrcoef': 0.9498126050346627}
    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9601409160538914, 'matthews_corrcoef': 0.9498126050346627}
    
    ------------------------------------------------------------------- Raw Metric Records -------------------------------------------------------------------
    ╒═════════════════════════╤══════════════════════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                                    │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪══════════════════════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M                              │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M                      │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M                   │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                         │ [0.7503] │  0.7503   │  0.7503  │   0   │   0   │ 0.7503 │ 0.7503 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M         │ [0.6602] │  0.6602   │  0.6602  │   0   │   0   │ 0.6602 │ 0.6602 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M                │ [0.7427] │  0.7427   │  0.7427  │   0   │   0   │ 0.7427 │ 0.7427 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-Region-Classification.Arabidopsis-OmniGenome-52M │ [0.9601] │  0.9601   │  0.9601  │   0   │   0   │ 0.9601 │ 0.9601 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M                              │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M                      │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M                   │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                         │ [0.6064] │  0.6064   │  0.6064  │   0   │   0   │ 0.6064 │ 0.6064 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M         │ [0.3295] │  0.3295   │  0.3295  │   0   │   0   │ 0.3295 │ 0.3295 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M                │ [0.495]  │   0.495   │  0.495   │   0   │   0   │ 0.495  │ 0.495  │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-Region-Classification.Arabidopsis-OmniGenome-52M │ [0.9498] │  0.9498   │  0.9498  │   0   │   0   │ 0.9498 │ 0.9498 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M                              │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M                              │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧══════════════════════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ---------------------------------------------------- https://github.com/yangheng95/metric_visualizer ----------------------------------------------------
    
    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    Running evaluation for task: RNA-Region-Classification.Rice Progress:  10 / 10 100.0%
    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  Loaded config for RNA-Region-Classification.Rice from __OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice\config.py
    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  {'task_name': 'RNA-Region-Classification.Rice', 'task_type': 'token_classification', 'label2id': {'3utr': 0, 'cds': 1, '5utr': 2}, 'num_labels': None, 'epochs': 50, 'patience': 5, 'learning_rate': 2e-05, 'weight_decay': 0, 'batch_size': 4, 'max_length': 1024, 'seeds': [45, 46, 47], 'compute_metrics': [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC06340>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7D9F50FE0>], 'train_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Rice/train.json', 'test_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Rice/test.json', 'valid_file': 'D:\\OneDrive - University of Exeter\\AIProjects\\OmniGenomeBench\\examples\\tutorials\\__OMNIGENOME_DATA__/benchmarks/RGB\\RNA-Region-Classification\\Rice/valid.json', 'dataset_cls': <class 'config.RegionClassificationDataset'>, 'model_cls': <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>}
    [2025-04-19 18:11:47] [OmniGenome 0.2.4alpha4]  Override epochs with 1 according to the input kwargs
    [2025-04-19 18:11:48] [OmniGenome 0.2.4alpha4]  Override batch_size with 4 according to the input kwargs
    [2025-04-19 18:11:48] [OmniGenome 0.2.4alpha4]  Override seeds with [42] according to the input kwargs
    [2025-04-19 18:11:48] [OmniGenome 0.2.4alpha4]  AutoBench Config for RNA-Region-Classification.Rice: task_name: RNA-Region-Classification.Rice
    task_type: token_classification
    label2id: {'3utr': 0, 'cds': 1, '5utr': 2}
    num_labels: None
    epochs: 1
    patience: 5
    learning_rate: 2e-05
    weight_decay: 0
    batch_size: 4
    max_length: 1024
    seeds: [42]
    compute_metrics: [<function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7DBC06340>, <function ClassificationMetric.__getattribute__.<locals>.wrapper at 0x000001F7D9F50FE0>]
    train_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/train.json
    test_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/test.json
    valid_file: D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/valid.json
    dataset_cls: <class 'config.RegionClassificationDataset'>
    model_cls: <class 'omnigenome.src.model.classiifcation.model.OmniGenomeModelForTokenClassification'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_tokenizer.py:41: UserWarning: No tokenizer wrapper found in anonymous8/OmniGenome-52M/omnigenome_wrapper.py -> Exception: Cannot find the module OmniGenomeTokenizerWrapper from anonymous8/OmniGenome-52M/omnigenome_wrapper.py.
      warnings.warn(
    Some weights of OmniGenomeModel were not initialized from the model checkpoint at anonymous8/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. parsed-literal::

    [2025-04-19 18:11:52] [OmniGenome 0.2.4alpha4]  Model Name: OmniGenomeModelForTokenClassification
    Model Metadata: {'library_name': 'OmniGenome', 'omnigenome_version': '0.2.4alpha4', 'torch_version': '2.5.1+cu12.4+gita8d6afb511a69687bbb2b7e88a3cf67917e1697e', 'transformers_version': '4.49.0', 'model_cls': 'OmniGenomeModelForTokenClassification', 'tokenizer_cls': 'EsmTokenizer', 'model_name': 'OmniGenomeModelForTokenClassification'}
    Base Model Name: anonymous8/OmniGenome-52M
    Model Type: omnigenome
    Model Architecture: None
    Model Parameters: 52.453345 M
    Model Config: OmniGenomeConfig {
      "OmniGenomefold_config": null,
      "_name_or_path": "anonymous8/OmniGenome-52M",
      "attention_probs_dropout_prob": 0.0,
      "auto_map": {
        "AutoConfig": "anonymous8/OmniGenome-52M--configuration_omnigenome.OmniGenomeConfig",
        "AutoModel": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeModel",
        "AutoModelForMaskedLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForMaskedLM",
        "AutoModelForSeq2SeqLM": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSeq2SeqLM",
        "AutoModelForSequenceClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForSequenceClassification",
        "AutoModelForTokenClassification": "anonymous8/OmniGenome-52M--modeling_omnigenome.OmniGenomeForTokenClassification"
      },
      "classifier_dropout": null,
      "emb_layer_norm_before": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0,
      "hidden_size": 480,
      "id2label": {
        "0": "3utr",
        "1": "cds",
        "2": "5utr"
      },
      "initializer_range": 0.02,
      "intermediate_size": 2400,
      "is_folding_model": false,
      "label2id": {
        "3utr": 0,
        "5utr": 2,
        "cds": 1
      },
      "layer_norm_eps": 1e-05,
      "mask_token_id": 23,
      "max_position_embeddings": 1026,
      "model_type": "omnigenome",
      "num_attention_heads": 24,
      "num_generation": 50,
      "num_hidden_layers": 16,
      "num_population": 100,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "token_dropout": true,
      "torch_dtype": "float32",
      "transformers_version": "4.49.0",
      "use_cache": true,
      "verify_ss": true,
      "vocab_list": null,
      "vocab_size": 24
    }
    
    
    [2025-04-19 18:11:52] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 18:11:52] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/train.json...
    [2025-04-19 18:11:52] [OmniGenome 0.2.4alpha4]  Loaded 21988 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/train.json
    [2025-04-19 18:11:52] [OmniGenome 0.2.4alpha4]  Detected shuffle=True, shuffling the examples...
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\abc\abstract_dataset.py:353: UserWarning: The 'sequence' field is missing in the raw dataset.
      warnings.warn("The 'sequence' field is missing in the raw dataset.")
    100%|██████████| 21988/21988 [03:22<00:00, 108.65it/s]
    

.. parsed-literal::

    [2025-04-19 18:15:15] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 998.5409314171367, 'max_seq_len': 1024, 'min_seq_len': 183, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 7,  ..., 7, 7, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 7,  ..., 4, 7, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 18:15:16] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/test.json...
    [2025-04-19 18:15:17] [OmniGenome 0.2.4alpha4]  Loaded 2749 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/test.json
    

.. parsed-literal::

    100%|██████████| 2749/2749 [00:25<00:00, 109.44it/s]
    

.. parsed-literal::

    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 998.8988723172063, 'max_seq_len': 1024, 'min_seq_len': 415, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 6, 5,  ..., 4, 4, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 7,  ..., 6, 4, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    2,    2, -100])}
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  Detected max_length=1024 in the dataset, using it as the max_length.
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  Loading data from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/valid.json...
    [2025-04-19 18:15:42] [OmniGenome 0.2.4alpha4]  Loaded 2749 examples from D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\examples\tutorials\__OMNIGENOME_DATA__/benchmarks/RGB\RNA-Region-Classification\Rice/valid.json
    

.. parsed-literal::

    100%|██████████| 2749/2749 [00:25<00:00, 107.16it/s]
    

.. parsed-literal::

    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  Max sequence length updated -> Reset max_length=1024, label_padding_length=1024
    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  {'avg_seq_len': 998.3728628592215, 'max_seq_len': 1024, 'min_seq_len': 335, 'avg_label_len': 1024.0, 'max_label_len': 1024, 'min_label_len': 1024}
    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  Preview of the first two samples in the dataset:
    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 7, 7,  ..., 6, 7, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  {'input_ids': tensor([0, 4, 6,  ..., 4, 6, 2]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([-100,    0,    0,  ...,    1,    1, -100])}
    [2025-04-19 18:16:08] [OmniGenome 0.2.4alpha4]  Using Trainer: <class 'omnigenome.src.trainer.accelerate_trainer.AccelerateTrainer'>
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:134: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler()
    Evaluating: 100%|██████████| 688/688 [00:42<00:00, 16.08it/s]
    

.. parsed-literal::

    [2025-04-19 18:16:54] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.30871424625722743, 'matthews_corrcoef': 0.02311674133518035}
    [2025-04-19 18:16:54] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.30871424625722743, 'matthews_corrcoef': 0.02311674133518035}
    

.. parsed-literal::

    Epoch 1/1 Loss: 0.6063: 100%|██████████| 5497/5497 [16:22<00:00,  5.60it/s]
    Evaluating: 100%|██████████| 688/688 [00:42<00:00, 16.16it/s]
    

.. parsed-literal::

    [2025-04-19 18:34:01] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9361362634555904, 'matthews_corrcoef': 0.9142724052009094}
    [2025-04-19 18:34:01] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9361362634555904, 'matthews_corrcoef': 0.9142724052009094}
    

.. parsed-literal::

    D:\OneDrive - University of Exeter\AIProjects\OmniGenomeBench\omnigenome\src\trainer\trainer.py:376: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      self.unwrap_model().load_state_dict(torch.load(self._model_state_dict_path))
    Testing: 100%|██████████| 688/688 [00:42<00:00, 16.15it/s]
    

.. parsed-literal::

    [2025-04-19 18:34:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9343617175084818, 'matthews_corrcoef': 0.918299054748215}
    [2025-04-19 18:34:46] [OmniGenome 0.2.4alpha4]  {'f1_score': 0.9343617175084818, 'matthews_corrcoef': 0.918299054748215}
    
    ------------------------------------------------------------------- Raw Metric Records -------------------------------------------------------------------
    ╒═════════════════════════╤══════════════════════════════════════════════════════════╤══════════╤═══════════╤══════════╤═══════╤═══════╤════════╤════════╕
    │ Metric                  │ Trial                                                    │ Values   │  Average  │  Median  │  Std  │  IQR  │  Min   │  Max   │
    ╞═════════════════════════╪══════════════════════════════════════════════════════════╪══════════╪═══════════╪══════════╪═══════╪═══════╪════════╪════════╡
    │ f1_score                │ RGB-RNA-SNMR-OmniGenome-52M                              │ [0.4705] │  0.4705   │  0.4705  │   0   │   0   │ 0.4705 │ 0.4705 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-Archive2-OmniGenome-52M                      │ [0.8801] │  0.8801   │  0.8801  │   0   │   0   │ 0.8801 │ 0.8801 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-rnastralign-OmniGenome-52M                   │ [0.9729] │  0.9729   │  0.9729  │   0   │   0   │ 0.9729 │ 0.9729 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                         │ [0.7503] │  0.7503   │  0.7503  │   0   │   0   │ 0.7503 │ 0.7503 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M         │ [0.6602] │  0.6602   │  0.6602  │   0   │   0   │ 0.6602 │ 0.6602 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M                │ [0.7427] │  0.7427   │  0.7427  │   0   │   0   │ 0.7427 │ 0.7427 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-Region-Classification.Arabidopsis-OmniGenome-52M │ [0.9601] │  0.9601   │  0.9601  │   0   │   0   │ 0.9601 │ 0.9601 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ f1_score                │ RGB-RNA-Region-Classification.Rice-OmniGenome-52M        │ [0.9344] │  0.9344   │  0.9344  │   0   │   0   │ 0.9344 │ 0.9344 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SNMR-OmniGenome-52M                              │ [0.3114] │  0.3114   │  0.3114  │   0   │   0   │ 0.3114 │ 0.3114 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-Archive2-OmniGenome-52M                      │ [0.8133] │  0.8133   │  0.8133  │   0   │   0   │ 0.8133 │ 0.8133 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-rnastralign-OmniGenome-52M                   │ [0.9579] │  0.9579   │  0.9579  │   0   │   0   │ 0.9579 │ 0.9579 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-SSP-bpRNA-OmniGenome-52M                         │ [0.6064] │  0.6064   │  0.6064  │   0   │   0   │ 0.6064 │ 0.6064 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Arabidopsis-OmniGenome-52M         │ [0.3295] │  0.3295   │  0.3295  │   0   │   0   │ 0.3295 │ 0.3295 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-TE-Prediction.Rice-OmniGenome-52M                │ [0.495]  │   0.495   │  0.495   │   0   │   0   │ 0.495  │ 0.495  │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-Region-Classification.Arabidopsis-OmniGenome-52M │ [0.9498] │  0.9498   │  0.9498  │   0   │   0   │ 0.9498 │ 0.9498 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ matthews_corrcoef       │ RGB-RNA-Region-Classification.Rice-OmniGenome-52M        │ [0.9183] │  0.9183   │  0.9183  │   0   │   0   │ 0.9183 │ 0.9183 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ roc_auc_score           │ RGB-RNA-SNMD-OmniGenome-52M                              │ [0.5971] │  0.5971   │  0.5971  │   0   │   0   │ 0.5971 │ 0.5971 │
    ├─────────────────────────┼──────────────────────────────────────────────────────────┼──────────┼───────────┼──────────┼───────┼───────┼────────┼────────┤
    │ root_mean_squared_error │ RGB-RNA-mRNA-OmniGenome-52M                              │ [0.7457] │  0.7457   │  0.7457  │   0   │   0   │ 0.7457 │ 0.7457 │
    ╘═════════════════════════╧══════════════════════════════════════════════════════════╧══════════╧═══════════╧══════════╧═══════╧═══════╧════════╧════════╛
    ---------------------------------------------------- https://github.com/yangheng95/metric_visualizer ----------------------------------------------------
    
    

5. Benchmark Checkpointing
--------------------------

Whenever the benchmark is interrupted, the results will be saved and
available for further execution. You can also clear the checkpoint to
start fresh:

.. code:: python

   AutoBench(bench_root=root, model_name_or_path=model_name_or_path, device=device, overwrite=True).run()
