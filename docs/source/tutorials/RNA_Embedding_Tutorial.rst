RNA Embedding
============================

This tutorial will guide you through how to use the RNA embedding model
using the ``OmniGenomeModelForEmbedding`` class. We will cover
initializing the model, encoding RNA sequences, saving/loading
embeddings, and computing similarities.

Step 1: Install Required Dependencies
-------------------------------------

Before we start, make sure you have the necessary libraries installed.
You can install them using the following command:

.. code:: ipython3

    !pip install OmniGenome torch transformers autocuda


.. parsed-literal::

    Requirement already satisfied: OmniGenome in d:\onedrive - university of exeter\aiprojects\omnigenomebench (0.2.5a0)
    Requirement already satisfied: torch in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (2.7.1+cu128)
    Requirement already satisfied: transformers in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (4.46.2)
    Requirement already satisfied: autocuda in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (0.16)
    Requirement already satisfied: findfile>=2.0.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (2.0.1)
    Requirement already satisfied: metric-visualizer>=0.9.6 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (0.9.17)
    Requirement already satisfied: termcolor in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (3.1.0)
    Requirement already satisfied: gitpython in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (3.1.44)
    Requirement already satisfied: pandas in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (2.3.0)
    Requirement already satisfied: viennarna in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (2.7.0)
    Requirement already satisfied: scikit-learn in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (1.7.0)
    Requirement already satisfied: accelerate in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (1.7.0)
    Requirement already satisfied: packaging in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from OmniGenome) (24.2)
    Requirement already satisfied: filelock in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (3.18.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy>=1.13.3 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (1.13.3)
    Requirement already satisfied: networkx in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (2025.5.1)
    Requirement already satisfied: setuptools in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from torch) (78.1.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (0.32.5)
    Requirement already satisfied: numpy>=1.17 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (2.3.0)
    Requirement already satisfied: pyyaml>=5.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (2024.11.6)
    Requirement already satisfied: requests in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (2.32.3)
    Requirement already satisfied: safetensors>=0.4.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (0.5.3)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (0.20.3)
    Requirement already satisfied: tqdm>=4.27 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from transformers) (4.67.1)
    Requirement already satisfied: matplotlib>=3.6.3 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.10.3)
    Requirement already satisfied: tikzplotlib in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.10.1)
    Requirement already satisfied: scipy in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (1.15.3)
    Requirement already satisfied: tabulate in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.9.0)
    Requirement already satisfied: natsort in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (8.4.0)
    Requirement already satisfied: update-checker in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.18.0)
    Requirement already satisfied: click in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (8.2.1)
    Requirement already satisfied: openpyxl in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.1.5)
    Requirement already satisfied: xlsxwriter in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.2.3)
    Requirement already satisfied: colorama in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.4.6)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (4.58.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (1.4.8)
    Requirement already satisfied: pillow>=8 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (11.2.1)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (1.17.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: psutil in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from accelerate->OmniGenome) (5.9.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from gitpython->OmniGenome) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from gitdb<5,>=4.0.1->gitpython->OmniGenome) (5.0.2)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: et-xmlfile in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from openpyxl->metric-visualizer>=0.9.6->OmniGenome) (2.0.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from pandas->OmniGenome) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from pandas->OmniGenome) (2025.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from requests->transformers) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from requests->transformers) (2025.4.26)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from scikit-learn->OmniGenome) (1.5.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from scikit-learn->OmniGenome) (3.6.0)
    Requirement already satisfied: webcolors in c:\users\hengu\miniconda3\envs\py312\lib\site-packages (from tikzplotlib->metric-visualizer>=0.9.6->OmniGenome) (24.11.1)
    

.. parsed-literal::

    WARNING: Ignoring invalid distribution ~orch (C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~orch (C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~orch (C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages)
    

Step 2: Setting Up the Embedding Model
--------------------------------------

First, let’s initialize the ``OmniGenomeModelForEmbedding`` class with a
pre-trained model.

.. code:: ipython3

    from omnigenome import OmniGenomeModelForEmbedding
    import torch
    
    # Initialize the model using a pre-trained model path (replace with RNA-specific model if available)
    model_name = "yangheng/OmniGenome-52M"  # Example model, replace with your own model
    embedding_model = OmniGenomeModelForEmbedding(model_name, trust_remote_code=True).to(torch.device("cuda:0")).to(torch.float16)


.. parsed-literal::

    C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages\requests\__init__.py:86: RequestsDependencyWarning: Unable to find acceptable character detection dependency (chardet or charset_normalizer).
      warnings.warn(
    C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
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
    
    

.. parsed-literal::

    Some weights of OmniGenomeModel were not initialized from the model checkpoint at yangheng/OmniGenome-52M and are newly initialized: ['OmniGenome.pooler.dense.bias', 'OmniGenome.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

Step 3: Encoding RNA Sequences into Embeddings
----------------------------------------------

We’ll now encode a batch of RNA sequences into embeddings.

.. code:: ipython3

    # Example RNA sequences (replace these with your own RNA sequences)
    rna_sequences = [
        "AUGGCUACG",
        "CGGAUACGGC",
        "UGGCCAAGUC",
        "AUGCUGCUAUGCUA"
    ]
    # Encode the RNA sequences into embeddings
    rna_embeddings = embedding_model.batch_encode(rna_sequences, agg='mean')
    
    # Display the generated embeddings
    print("RNA Embeddings:")
    print(rna_embeddings)


.. parsed-literal::

    [2025-06-16 22:58:40] [OmniGenome 0.2.6alpha0]  Generated embeddings for 4 sequences.
    RNA Embeddings:
    tensor([[-0.4038, -1.0078, -0.0919,  ..., -0.6841, -0.9468, -0.2502],
            [-0.2445, -0.7437, -0.2668,  ..., -0.2125, -0.9575, -0.1359],
            [-0.4094, -0.8535, -0.0769,  ..., -0.5132, -0.5581, -0.3665],
            [-0.3696, -0.7798, -0.0314,  ..., -0.6567, -1.0420, -0.0429]],
           dtype=torch.float16)
    

Step 4: Saving and Loading Embeddings
-------------------------------------

You can save the generated embeddings to a file and load them later when
needed.

.. code:: ipython3

    # Save embeddings to a file
    embedding_model.save_embeddings(rna_embeddings, "rna_embeddings.pt")
    
    # Load the embeddings from the file
    loaded_embeddings = embedding_model.load_embeddings("rna_embeddings.pt")
    
    # Display the loaded embeddings to verify
    print("Loaded RNA Embeddings:")
    print(loaded_embeddings)


.. parsed-literal::

    [2025-06-16 22:58:40] [OmniGenome 0.2.6alpha0]  Embeddings saved to rna_embeddings.pt
    [2025-06-16 22:58:40] [OmniGenome 0.2.6alpha0]  Loaded embeddings from rna_embeddings.pt
    Loaded RNA Embeddings:
    tensor([[-0.4038, -1.0078, -0.0919,  ..., -0.6841, -0.9468, -0.2502],
            [-0.2445, -0.7437, -0.2668,  ..., -0.2125, -0.9575, -0.1359],
            [-0.4094, -0.8535, -0.0769,  ..., -0.5132, -0.5581, -0.3665],
            [-0.3696, -0.7798, -0.0314,  ..., -0.6567, -1.0420, -0.0429]],
           dtype=torch.float16)
    

Step 5: Computing Similarity Between RNA Sequences
--------------------------------------------------

Let’s compute the similarity between two RNA sequence embeddings using
cosine similarity.

.. code:: ipython3

    # Compute the similarity between the first two RNA sequence embeddings
    similarity = embedding_model.compute_similarity(loaded_embeddings[0], loaded_embeddings[1])
    
    # Display the similarity score
    print(f"Similarity between the first two RNA sequences: {similarity:.4f}")


.. parsed-literal::

    Similarity between the first two RNA sequences: 0.9395
    

Step 6: Encoding a Single RNA Sequence
--------------------------------------

You can also encode a single RNA sequence into its embedding.

.. code:: ipython3

    # Example single RNA sequence
    single_rna_sequence = "AUGGCUACG"
    
    # Get the embedding for the single RNA sequence
    
    head_rna_embedding = embedding_model.encode(rna_sequences[0], agg='head', keep_dim=True)  # Encode a single RNA sequence
    mean_rna_embedding = embedding_model.encode(rna_sequences[0], agg='mean')  # Encode a single RNA sequence
    tail_rna_embedding = embedding_model.encode(rna_sequences[0], agg='tail')  # Encode a single RNA sequence
    
    # Display the embedding for the single RNA sequence
    print("Single RNA Sequence Embedding:")
    print(head_rna_embedding)


.. parsed-literal::

    C:\Users\hengu\miniconda3\envs\py312\Lib\site-packages\executing\executing.py:713: DeprecationWarning: ast.Str is deprecated and will be removed in Python 3.14; use ast.Constant instead
      right=ast.Str(s=sentinel),
    C:\Users\hengu\miniconda3\envs\py312\Lib\ast.py:587: DeprecationWarning: Attribute s is deprecated and will be removed in Python 3.14; use value instead
      return Constant(*args, **kwargs)
    

::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[6], line 5
          2 single_rna_sequence = "AUGGCUACG"
          4 # Get the embedding for the single RNA sequence
    ----> 5 single_rna_embedding = embedding_model.encode_single_sequence(single_rna_sequence)
          7 # Display the embedding for the single RNA sequence
          8 print("Single RNA Sequence Embedding:")
    

    File ~\miniconda3\envs\py312\Lib\site-packages\torch\nn\modules\module.py:1940, in Module.__getattr__(self, name)
       1938     if name in modules:
       1939         return modules[name]
    -> 1940 raise AttributeError(
       1941     f"'{type(self).__name__}' object has no attribute '{name}'"
       1942 )
    

    AttributeError: 'OmniGenomeModelForEmbedding' object has no attribute 'encode_single_sequence'


Full Example
------------

Here’s a complete example that walks through all the steps we covered in
the tutorial.

.. code:: ipython3

    from omnigenome import OmniGenomeModelForEmbedding
    
    # Step 1: Initialize the model
    model_name = "yangheng/OmniGenome-52M"  # Replace with your RNA-specific model
    embedding_model = OmniGenomeModelForEmbedding(model_name)
    
    # Step 2: Encode RNA sequences
    rna_sequences = ["AUGGCUACG", "CGGAUACGGC"]
    rna_embeddings = embedding_model.encode_sequences(rna_sequences)
    print("RNA Embeddings:", rna_embeddings)
    
    # Step 3: Save embeddings to a file
    embedding_model.save_embeddings(rna_embeddings, "rna_embeddings.pt")
    
    # Step 4: Load embeddings from the file
    loaded_embeddings = embedding_model.load_embeddings("rna_embeddings.pt")
    
    # Step 5: Compute similarity between the first two RNA sequence embeddings
    similarity = embedding_model.compute_similarity(loaded_embeddings[0], loaded_embeddings[1])
    print(f"Similarity between RNA sequences: {similarity:.4f}")
    
    # Step 6: Encode a single RNA sequence
    single_rna_sequence = "AUGGCUACG"
    single_rna_embedding = embedding_model.encode_single_sequence(single_rna_sequence)
    print("Single RNA Sequence Embedding:", single_rna_embedding)
