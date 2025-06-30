.. OmniGenBench documentation master file, created by
   sphinx-quickstart on Thu Jun 26 20:19:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OmniGenBench's documentation!
============================================

`OmniGenBench <https://github.com/COLA-Laboratory/OmniGenBench>`_ is a modular benchmarking platform for evaluating genomic foundation models (GFMs) across diverse tasks like RNA structure prediction, gene function classification, and multi-species generalization.


Installation
============================================
Before installing OmniGenome, you need to install the following dependencies:

- Python 3.10+
- PyTorch 2.5+
- Transformers 4.46.0+


**PyPI Installation**

Install the latest stable version from `PyPI <https://pypi.org/project/OmniGenome/>`_:

.. code-block:: bash

   pip install spikingjelly omnigenome -U


**Source Installation**

Or you can clone the repository and install it from source:

.. code-block:: bash

   git clone https://github.com/yangheng95/OmniGenBench.git
   cd OmniGenBench
   pip install -e .


Supported Models
============================================

OmniGenBench provides plug-and-play evaluation for over **30 genomic foundation models**, covering both **RNA** and **DNA** modalities. The following are highlights:

+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| Model          | Params | Pre-training Corpus                        | Highlights                                          |
+================+========+============================================+=====================================================+
| **OmniGenome** | 186M   | 54B plant RNA+DNA tokens                   | Multi-modal, structure-aware encoder                |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **Agro-NT-1B** | 985M   | 48 edible-plant genomes                    | Billion-scale DNA LM w/ NT-V2 k-mer vocab           |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **RiNALMo**    | 651M   | 36M ncRNA sequences                        | Largest public RNA LM; FlashAttention-2             |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **DNABERT-2**  | 117M   | 32B DNA tokens, 136 species (BPE)          | Byte-pair encoding; 2nd-gen DNA BERT                |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **RNA-FM**     | 96M    | 23M ncRNA sequences                        | High performance on RNA structure tasks             |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **RNA-MSM**    | 96M    | Multi-sequence alignments                  | MSA-based evolutionary RNA LM                       |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **NT-V2**      | 96M    | 300B DNA tokens (850 species)              | Hybrid k-mer vocabulary                             |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **HyenaDNA**   | 47M    | Human chromosomes                          | Long-context autoregressive model (1Mb)             |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **SpliceBERT** | 19M    | 2M pre-mRNA sequences                      | Fine-grained splice-site recognition                |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **Caduceus**   | 1.9M   | Human chromosomes                          | Ultra-compact DNA LM (RC-equivariant)               |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| **RNA-BERT**   | 0.5M   | 4,000+ ncRNA families                      | Small BERT with nucleotide masking                  |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+
| *...and more*  | —      | See Appendix E of the paper                | Includes PlantRNA-FM, UTR-LM, MP-RNA, CALM, etc.    |
+----------------+--------+--------------------------------------------+-----------------------------------------------------+


Benchmarks
============================================

OmniGenBench supports five curated benchmark suites covering both **sequence-level** and **structure-level** genomics tasks across species.

+--------------+-----------------------------+--------------------------+------------------------------------------------------+
| Suite        | Focus                       | #Tasks / Datasets        | Sample Tasks                                         |
+==============+=============================+==========================+======================================================+
| **RGB**      | RNA structure + function    | 12 tasks (SN-level)      | RNA secondary structure, SNMR, degradation prediction|
+--------------+-----------------------------+--------------------------+------------------------------------------------------+
| **BEACON**   | RNA (multi-domain)          | 13 tasks                 | Base pairing, mRNA design, RNA contact maps          |
+--------------+-----------------------------+--------------------------+------------------------------------------------------+
| **PGB**      | Plant long-range DNA        | 7 categories             | PolyA, enhancer, chromatin access, splice site       |
+--------------+-----------------------------+--------------------------+------------------------------------------------------+
| **GUE**      | DNA general tasks           | 36 datasets (9 tasks)    | TF binding, core promoter, enhancer detection        |
+--------------+-----------------------------+--------------------------+------------------------------------------------------+
| **GB**       | Classic DNA classification  | 9 datasets               | Human/mouse enhancer, promoter variant classification|
+--------------+-----------------------------+--------------------------+------------------------------------------------------+





.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS:

   tutorials/AutoBench_Tutorial.rst
   tutorials/RNA_Augmentation_Tutorial.rst
   tutorials/RNA_Design_Tutorial.rst
   tutorials/RNA_Embedding_Tutorial.rst
   tutorials/Secondary_Structure_Prediction_Tutorial.rst
   tutorials/ZeroShot_Structure_Prediction_Tutorial.rst


Modules Docs
============================================
.. toctree::
   :maxdepth: 1
   :caption: APIs

   omnigenome.auto
   omnigenome.cli
   omnigenome.src
   omnigenome.utility


Citation
============================================
If you use OmniGenBench in you work, please cite it as follows:

.. code-block:: bash

   @article{
   doi:10.1126/sciadv.adi1480,
   author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
   title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
   journal = {Science Advances},
   volume = {9},
   number = {40},
   pages = {eadi1480},
   year = {2023},
   doi = {10.1126/sciadv.adi1480},
   URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
   eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
   abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}

License
============================================
OmniGenomeBench is licensed under the Apache License 2.0. See the LICENSE file for more information.

Contribution
============================================
We welcome contributions to OmniGenomeBench! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request on GitHub.



