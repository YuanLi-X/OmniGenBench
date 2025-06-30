Sequence Augmentation
========================================================================

This tutorial will guide you through the steps of using the
``OmniGenomeModelForAugmentation`` class to augment RNA/DNA sequences by
adding noise and using a masked language model (MLM) to fill in the
masked tokens. The model allows you to generate multiple augmented
instances for each sequence, and you can configure the noise ratio,
maximum token length, and number of instances.

1. **Setting Up the Environment**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting, you need to install the required Python packages:

.. code:: bash

   pip install torch transformers autocuda tqdm

You will also need a pre-trained masked language model (MLM) that is
compatible with your sequence data. The model should be hosted on
Hugging Face or available locally.

2. **Understanding the Parameters**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When initializing the ``OmniGenomeModelForAugmentation`` class, you can
configure several key parameters: - ``model_name_or_path``: The Hugging
Face model name or the local path to the pre-trained model. -
``noise_ratio``: The proportion of tokens to mask in each sequence for
augmentation (default is 0.15). - ``max_length``: The maximum token
length for input sequences (default is 1026). - ``instance_num``: The
number of augmented instances to generate for each input sequence
(default is 1).

3. **Example Usage**
~~~~~~~~~~~~~~~~~~~~

Let’s walk through an example of how to use the
``OmniGenomeModelForAugmentation`` class.

First, initialize the model by providing the model path and other
augmentation parameters such as noise ratio, maximum sequence length,
and instance number.

.. code:: ipython3

    from OmniGenomeModelForAugmentation import OmniGenomeModelForAugmentation
    
    # Initialize the augmentation model
    model = OmniGenomeModelForAugmentation(
        model_name_or_path="anonymous8/OmniGenome-186M",  # Pre-trained model
        noise_ratio=0.2,  # 20% of the tokens will be masked
        max_length=1026,  # Maximum token length
        instance_num=3  # Generate 3 augmented instances per sequence
    )
    

Step 1: **Augment a Single Sequence**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can augment a single sequence directly by calling the
``augment_sequence`` method. This method will apply noise, predict
masked tokens, and return the augmented sequence.

.. code:: ipython3

    # Test single sequence augmentation
    augmented_sequence = model.augment_sequence("ATCTTGCATTGAAG")
    print(f"Augmented sequence: {augmented_sequence}")

Step 2: **Augment Sequences from a File**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To augment multiple sequences from a JSON file, you can use the
``augment_from_file`` method. This method reads the sequences from the
file, applies augmentation, and saves the augmented sequences to another
file.

.. code:: ipython3

    # Define file paths for input and output
    input_file = "toy_datasets/test.json"
    output_file = "toy_datasets/augmented_sequences.json"
    
    # Augment sequences from the input file and save to the output file
    model.augment_from_file(input_file, output_file)
    

The input file should be in JSON format, where each line contains a
sequence, like this:

.. code:: json

   {"seq": "ATCTTGCATTGAAG"}
   {"seq": "GGTTTACAGTCCAA"}

The output will be saved in the same format, with each augmented
sequence written in a new line.

Step 3: **Configurable Parameters**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The augmentation process allows you to configure various parameters,
such as: - **``noise_ratio``**: Specifies the percentage of tokens that
will be masked in the input sequence. The default value is ``0.15``
(i.e., 15% of tokens will be masked). - **``max_length``**: The maximum
token length for the input sequences. The default is ``1026``. -
**``instance_num``**: The number of augmented instances to generate for
each input sequence. The default is ``1``, but you can increase this
value to create multiple augmented versions of each sequence.

Step 4: **Save Augmented Sequences**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``save_augmented_sequences`` method saves the generated augmented
sequences to a JSON file. Each line will contain one augmented sequence
in the format ``{"aug_seq": "<augmented_sequence>"}``.

Conclusion
~~~~~~~~~~

The ``OmniGenomeModelForAugmentation`` class provides a simple and
flexible interface for augmenting sequences using a masked language
model. By adjusting the noise ratio, instance count, and other
hyperparameters, you can create diverse augmented datasets to improve
the performance of machine learning models.
