RNA Design
===========================================



In this tutorial, we will walk through how to set up and use the
``OmniGenomeModelforRNADesign`` class to design RNA sequences. We will
cover the following topics: 1. Setting up the model 2. Running RNA
design 3. Saving and loading results 4. Fine-tuning the parameters 5.
Visualizing RNA structures

Tutorial 1: Setting Up the OmniGenome Model for RNA Design
----------------------------------------------------------

.. code:: ipython3

    
    # Install dependencies (run this if needed)
    !pip install OmniGenome torch transformers autocuda viennaRNA tqdm -U


.. parsed-literal::

    Requirement already satisfied: OmniGenome in c:\users\chuan\miniconda3\lib\site-packages (0.1.0a0)
    Collecting OmniGenome
      Downloading OmniGenome-0.1.1a0-py3-none-any.whl.metadata (3.9 kB)
    Requirement already satisfied: torch in c:\users\chuan\miniconda3\lib\site-packages (2.4.1)
    Requirement already satisfied: transformers in c:\users\chuan\miniconda3\lib\site-packages (4.44.2)
    Collecting transformers
      Using cached transformers-4.45.1-py3-none-any.whl.metadata (44 kB)
    Requirement already satisfied: autocuda in c:\users\chuan\miniconda3\lib\site-packages (0.16)
    Requirement already satisfied: viennaRNA in c:\users\chuan\miniconda3\lib\site-packages (2.6.4)
    Requirement already satisfied: tqdm in c:\users\chuan\miniconda3\lib\site-packages (4.66.5)
    Requirement already satisfied: findfile>=2.0.0 in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (2.0.1)
    Requirement already satisfied: metric-visualizer>=0.9.6 in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (0.9.13.post1)
    Requirement already satisfied: termcolor in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (2.4.0)
    Requirement already satisfied: gitpython in c:\users\chuan\appdata\roaming\python\python39\site-packages (from OmniGenome) (3.1.27)
    Requirement already satisfied: sentencepiece in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (0.1.99)
    Requirement already satisfied: protobuf<4.0.0 in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (3.20.3)
    Requirement already satisfied: pandas in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (2.1.4)
    Requirement already satisfied: scikit-learn in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (1.4.2)
    Requirement already satisfied: accelerate in c:\users\chuan\miniconda3\lib\site-packages (from OmniGenome) (0.33.0)
    Requirement already satisfied: filelock in c:\users\chuan\miniconda3\lib\site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in c:\users\chuan\miniconda3\lib\site-packages (from torch) (4.11.0)
    Requirement already satisfied: sympy in c:\users\chuan\appdata\roaming\python\python39\site-packages (from torch) (1.11.1)
    Requirement already satisfied: networkx in c:\users\chuan\miniconda3\lib\site-packages (from torch) (3.1)
    Requirement already satisfied: jinja2 in c:\users\chuan\miniconda3\lib\site-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in c:\users\chuan\miniconda3\lib\site-packages (from torch) (2023.10.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (0.24.2)
    Requirement already satisfied: numpy>=1.17 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (1.26.2)
    Requirement already satisfied: packaging>=20.0 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (23.1)
    Requirement already satisfied: pyyaml>=5.1 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (2023.10.3)
    Requirement already satisfied: requests in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (2.32.3)
    Requirement already satisfied: safetensors>=0.4.1 in c:\users\chuan\miniconda3\lib\site-packages (from transformers) (0.4.1)
    Collecting tokenizers<0.21,>=0.20 (from transformers)
      Using cached tokenizers-0.20.0-cp39-none-win_amd64.whl.metadata (6.9 kB)
    Requirement already satisfied: colorama in c:\users\chuan\miniconda3\lib\site-packages (from tqdm) (0.4.6)
    Requirement already satisfied: matplotlib>=3.6.3 in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.8.2)
    Requirement already satisfied: tikzplotlib in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.10.1)
    Requirement already satisfied: scipy>=1.10.0 in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (1.10.1)
    Requirement already satisfied: tabulate in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.9.0)
    Requirement already satisfied: natsort in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (8.4.0)
    Requirement already satisfied: update-checker in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (0.18.0)
    Requirement already satisfied: click in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (8.0.4)
    Requirement already satisfied: openpyxl in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.1.2)
    Requirement already satisfied: xlsxwriter in c:\users\chuan\miniconda3\lib\site-packages (from metric-visualizer>=0.9.6->OmniGenome) (3.1.9)
    Requirement already satisfied: psutil in c:\users\chuan\miniconda3\lib\site-packages (from accelerate->OmniGenome) (5.9.7)
    Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\chuan\miniconda3\lib\site-packages (from gitpython->OmniGenome) (4.0.11)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\chuan\miniconda3\lib\site-packages (from jinja2->torch) (2.1.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\chuan\miniconda3\lib\site-packages (from pandas->OmniGenome) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\chuan\miniconda3\lib\site-packages (from pandas->OmniGenome) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\chuan\miniconda3\lib\site-packages (from pandas->OmniGenome) (2023.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\chuan\miniconda3\lib\site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\chuan\miniconda3\lib\site-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\chuan\miniconda3\lib\site-packages (from requests->transformers) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\chuan\miniconda3\lib\site-packages (from requests->transformers) (2024.7.4)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\chuan\miniconda3\lib\site-packages (from scikit-learn->OmniGenome) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\chuan\miniconda3\lib\site-packages (from scikit-learn->OmniGenome) (3.2.0)
    Requirement already satisfied: mpmath>=0.19 in c:\users\chuan\appdata\roaming\python\python39\site-packages (from sympy->torch) (1.2.1)
    Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\chuan\miniconda3\lib\site-packages (from gitdb<5,>=4.0.1->gitpython->OmniGenome) (5.0.1)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (1.4.4)
    Requirement already satisfied: pillow>=8 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (10.0.1)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (3.0.9)
    Requirement already satisfied: importlib-resources>=3.2.0 in c:\users\chuan\miniconda3\lib\site-packages (from matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (6.1.1)
    Requirement already satisfied: six>=1.5 in c:\users\chuan\miniconda3\lib\site-packages (from python-dateutil>=2.8.2->pandas->OmniGenome) (1.16.0)
    Requirement already satisfied: et-xmlfile in c:\users\chuan\miniconda3\lib\site-packages (from openpyxl->metric-visualizer>=0.9.6->OmniGenome) (1.1.0)
    Requirement already satisfied: webcolors in c:\users\chuan\miniconda3\lib\site-packages (from tikzplotlib->metric-visualizer>=0.9.6->OmniGenome) (1.13)
    Requirement already satisfied: zipp>=3.1.0 in c:\users\chuan\miniconda3\lib\site-packages (from importlib-resources>=3.2.0->matplotlib>=3.6.3->metric-visualizer>=0.9.6->OmniGenome) (3.17.0)
    Downloading OmniGenome-0.1.1a0-py3-none-any.whl (118 kB)
       ---------------------------------------- 0.0/118.2 kB ? eta -:--:--
       --- ------------------------------------ 10.2/118.2 kB ? eta -:--:--
       ---------- ---------------------------- 30.7/118.2 kB 435.7 kB/s eta 0:00:01
       ---------------------------------------- 118.2/118.2 kB 1.1 MB/s eta 0:00:00
    Using cached transformers-4.45.1-py3-none-any.whl (9.9 MB)
    Using cached tokenizers-0.20.0-cp39-none-win_amd64.whl (2.3 MB)
    Installing collected packages: tokenizers, transformers, OmniGenome
      Attempting uninstall: tokenizers
        Found existing installation: tokenizers 0.19.1
        Uninstalling tokenizers-0.19.1:
          Successfully uninstalled tokenizers-0.19.1
      Attempting uninstall: transformers
        Found existing installation: transformers 4.44.2
        Uninstalling transformers-4.44.2:
          Successfully uninstalled transformers-4.44.2
      Attempting uninstall: OmniGenome
        Found existing installation: OmniGenome 0.1.0a0
        Uninstalling OmniGenome-0.1.0a0:
          Successfully uninstalled OmniGenome-0.1.0a0
    Successfully installed OmniGenome-0.1.1a0 tokenizers-0.20.0 transformers-4.45.1
    

.. parsed-literal::

    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -orch (c:\users\chuan\miniconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -orch (c:\users\chuan\miniconda3\lib\site-packages)
    DEPRECATION: pytorch-lightning 1.7.6 has a non-standard dependency specifier torch>=1.9.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    DEPRECATION: torchsde 0.2.5 has a non-standard dependency specifier numpy>=1.19.*; python_version >= "3.7". pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of torchsde or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
        WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
        WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    textattack 0.3.8 requires transformers==4.30.0, but you have transformers 4.45.1 which is incompatible.
    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -ensorflow-gpu (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\chuan\appdata\roaming\python\python39\site-packages)
    

.. code:: ipython3

    from omnigenome import OmniGenomeModelForRNADesign  # Assuming this is where the model class is defined
     
    # Initialize the model for RNA design
    model = OmniGenomeModelForRNADesign(model_path="anonymous8/OmniGenome-186M")

Explanation
~~~~~~~~~~~

- **model_path**: Path to the pre-trained model for RNA design. We are
  using ``"anonymous8/OmniGenome-186M"``.

Tutorial 2: Running RNA Sequence Design
---------------------------------------

.. code:: ipython3

    
    # Define the target RNA structure
    structure = "(((....)))"  # Example of a simple RNA hairpin structure
    
    # Run the genetic algorithm to design RNA sequences
    best_sequences = model.run_rna_design(structure=structure, mutation_ratio=0.5, num_population=100, num_generation=100)
    
    # Print the best sequence(s)
    print("Best RNA sequences:", best_sequences)
    


.. parsed-literal::

    Best RNA sequences: ['GCTGCTGGGC', 'GCTGTGGGGC', 'GCCAGCTGGC', 'GCTCTGGAGC', 'GCTGATGGGC', 'GGTGGCAGCC', 'GCCAAAGGGC', 'GCTGGAGGGC', 'GCCAAAGGGC', 'CGGATTCCCG', 'GCTCTCAAGC', 'GCTGTGGGGC', 'GGGCTTTCCC', 'GCTCAAGGGC', 'GCGCGCGCGC', 'CGCCTCGGCG', 'GCTGAGAGGC', 'GCTGCAGGGC', 'GCTGAAGGGC', 'GGCGAGGGCC', 'GCTAGGAGGC', 'GGGCTTGCCC', 'GGGATGGCCC', 'GCTGCCAAGC', 'GGCGAGGGCC', 'GCTGGCGGGC', 'GCCTTTTGGC', 'GGTGAAGGCC', 'GGCGGCGGCC', 'GCGGCTGCGC', 'GCTGCATGGC', 'GCTGTGGGGC', 'CGCGCGGGCG', 'GGTGCCCGCC', 'TGGAACCCCA', 'GCCCATGGGC', 'CCGAAGCCGG', 'GGGGGGGCCC', 'GCTGCATAGC', 'GCCCTCTGGC', 'GCCGCGGGGC', 'GCTACATGGC', 'GCGGGAGCGC', 'GGTGGCTGCC', 'GCCGTGGGGC', 'GCGCCCCCGC', 'GGTGTCAGCC', 'GGTGTGGGCC', 'GCTCCCGGGC', 'GCTGAGGAGC', 'GCTGCTGGGC', 'GGCCTTCGCC', 'GCGCCCCCGC', 'GCCCTTGGGC', 'GCCGTGGGGC', 'GGCGGCGGCC', 'CGTGCTGACG', 'CCTGAGGAGG', 'GCTACTTGGC', 'TGCGAGGGCA', 'GGCAAAGGCC', 'GCTGAAGAGC', 'CGGCTTGCCG', 'GGGCTTGCCC', 'GCTGAAGAGC', 'GCTGAAGGGC', 'GCCAGTGGGC', 'GGCGCGGGCC', 'GCGGAGGCGC', 'CCTGAGGGGG', 'GCGAAACCGC', 'GCTGAGGGGC', 'GCTTGCAGGC', 'GCTTTCTGGC', 'GGGCTGGCCC', 'GCCATGAGGC', 'GAGGAAGCTC', 'GCTGAAGAGC', 'GCTGCAAGGC', 'CGGGCGGCCG', 'GCCGCGGGGC', 'GCGCGCGCGC', 'CCTGAGGGGG', 'GGGGCTGCCC', 'GCTGAGAGGC', 'GCTAAATGGC', 'GCCGGCAGGC', 'GCCGCTGGGC', 'GCTGGAGGGC', 'GGCGGCGGCC']
    

In this tutorial, we: - Defined the RNA structure - Ran the genetic
algorithm for RNA design

Tutorial 3: Saving and Loading Designed RNA Sequences
-----------------------------------------------------

.. code:: ipython3

    
    import json
    
    # Save the best sequences to a file
    output_file = "best_rna_sequences.json"
    with open(output_file, "w") as f:
        json.dump({"structure": structure, "best_sequences": best_sequences}, f)
    
    print(f"Best sequences saved to {output_file}")
    


.. parsed-literal::

    Best sequences saved to best_rna_sequences.json
    

.. code:: ipython3

    
    # Load the sequences from the saved file
    with open(output_file, "r") as f:
        loaded_data = json.load(f)
    
    print("Loaded RNA structure:", loaded_data["structure"])
    print("Loaded best sequences:", loaded_data["best_sequences"])
    


.. parsed-literal::

    Loaded RNA structure: (((....)))
    Loaded best sequences: ['GCTGCTGGGC', 'GCTGTGGGGC', 'GCCAGCTGGC', 'GCTCTGGAGC', 'GCTGATGGGC', 'GGTGGCAGCC', 'GCCAAAGGGC', 'GCTGGAGGGC', 'GCCAAAGGGC', 'CGGATTCCCG', 'GCTCTCAAGC', 'GCTGTGGGGC', 'GGGCTTTCCC', 'GCTCAAGGGC', 'GCGCGCGCGC', 'CGCCTCGGCG', 'GCTGAGAGGC', 'GCTGCAGGGC', 'GCTGAAGGGC', 'GGCGAGGGCC', 'GCTAGGAGGC', 'GGGCTTGCCC', 'GGGATGGCCC', 'GCTGCCAAGC', 'GGCGAGGGCC', 'GCTGGCGGGC', 'GCCTTTTGGC', 'GGTGAAGGCC', 'GGCGGCGGCC', 'GCGGCTGCGC', 'GCTGCATGGC', 'GCTGTGGGGC', 'CGCGCGGGCG', 'GGTGCCCGCC', 'TGGAACCCCA', 'GCCCATGGGC', 'CCGAAGCCGG', 'GGGGGGGCCC', 'GCTGCATAGC', 'GCCCTCTGGC', 'GCCGCGGGGC', 'GCTACATGGC', 'GCGGGAGCGC', 'GGTGGCTGCC', 'GCCGTGGGGC', 'GCGCCCCCGC', 'GGTGTCAGCC', 'GGTGTGGGCC', 'GCTCCCGGGC', 'GCTGAGGAGC', 'GCTGCTGGGC', 'GGCCTTCGCC', 'GCGCCCCCGC', 'GCCCTTGGGC', 'GCCGTGGGGC', 'GGCGGCGGCC', 'CGTGCTGACG', 'CCTGAGGAGG', 'GCTACTTGGC', 'TGCGAGGGCA', 'GGCAAAGGCC', 'GCTGAAGAGC', 'CGGCTTGCCG', 'GGGCTTGCCC', 'GCTGAAGAGC', 'GCTGAAGGGC', 'GCCAGTGGGC', 'GGCGCGGGCC', 'GCGGAGGCGC', 'CCTGAGGGGG', 'GCGAAACCGC', 'GCTGAGGGGC', 'GCTTGCAGGC', 'GCTTTCTGGC', 'GGGCTGGCCC', 'GCCATGAGGC', 'GAGGAAGCTC', 'GCTGAAGAGC', 'GCTGCAAGGC', 'CGGGCGGCCG', 'GCCGCGGGGC', 'GCGCGCGCGC', 'CCTGAGGGGG', 'GGGGCTGCCC', 'GCTGAGAGGC', 'GCTAAATGGC', 'GCCGGCAGGC', 'GCCGCTGGGC', 'GCTGGAGGGC', 'GGCGGCGGCC']
    

Tutorial 4: Fine-Tuning Parameters for Better RNA Sequence Design
-----------------------------------------------------------------

.. code:: ipython3

    
    # Run the design with a higher mutation ratio
    best_sequences = model.run_rna_design(structure=structure, mutation_ratio=0.7, num_population=100, num_generation=100)
    print("Best RNA sequences with higher mutation:", best_sequences)
    

.. code:: ipython3

    
    # Run the design with a larger population size
    best_sequences = model.run_rna_design(structure=structure, mutation_ratio=0.5, num_population=200, num_generation=100)
    print("Best RNA sequences with larger population:", best_sequences)
    

.. code:: ipython3

    
    # Run the design for more generations
    best_sequences = model.run_rna_design(structure=structure, mutation_ratio=0.5, num_population=100, num_generation=200)
    print("Best RNA sequences with more generations:", best_sequences)
    

Tutorial 5: Visualizing the RNA Structure
-----------------------------------------

You can visualize the RNA secondary structure using external tools like
RNAfold from ViennaRNA.

Step 1: Install RNAfold
~~~~~~~~~~~~~~~~~~~~~~~

To install RNAfold, you can use the following command (if on Ubuntu):

.. code:: bash

   sudo apt-get install vienna-rna

Step 2: Visualizing the Designed RNA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After obtaining your RNA sequence, you can visualize its secondary
structure using RNAfold:

.. code:: bash

   echo "GCGCUACGUCGCGAU" | RNAfold

This will output the predicted secondary structure along with the
minimum free energy (MFE).

Conclusion
----------

By following these tutorials, you can: - Set up and initialize the
OmniGenomeModelforRNADesign for RNA sequence design. - Run RNA sequence
design with a genetic algorithm. - Tune the parameters to optimize the
design process. - Save and load results. - Visualize the RNA secondary
structure using RNAfold.

Explore more advanced configurations and tweak parameters for better
results!
