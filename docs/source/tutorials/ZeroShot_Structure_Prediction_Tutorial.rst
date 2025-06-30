Zero-Shot Secondary Structure Prediction
==================================================================

In this tutorial, you’ll learn how to use the ``OmniGenome`` model for
RNA secondary structure prediction in a zero-shot setting. The secondary
structure of RNA is essential for understanding its function and
interactions, and using machine learning models like ``OmniGenome``
allows us to make accurate predictions directly from sequence data.

We will demonstrate how to: - Load a pre-trained model for RNA secondary
structure prediction. - Perform zero-shot prediction on an RNA sequence.
- Use a simplified API for folding sequences. - Compare predictions with
the popular RNA folding tool ``ViennaRNA``.

1. **Setting Up the Environment**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you have the required Python packages installed. You can
install them by running:

.. code:: bash

   pip install torch transformers autocuda viennarna sklearn

We’ll also be using the pre-trained ``OmniGenome`` model from Hugging
Face for token classification. The specific model,
``anonymous8/OmniGenome-186M``, is trained on RNA secondary structure
prediction tasks.

2. **Loading the Model and Tokenizer**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to load the pre-trained model and tokenizer for RNA
secondary structure prediction.

.. code:: ipython3

    import torch
    import autocuda
    from transformers import OmniGenomeForTokenClassification, AutoTokenizer
    
    # Load the pre-trained model for secondary structure prediction
    ssp_model = OmniGenomeForTokenClassification.from_pretrained(
        "anonymous8/OmniGenome-186M"
    ).to(autocuda.auto_cuda())
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("anonymous8/OmniGenome-186M")
    

3. **Defining the Prediction Function**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we define a function ``zero_shot_secondary_structure_prediction``
that takes a model and an RNA sequence as input and outputs the
predicted secondary structure.

This function works by tokenizing the input sequence, passing it through
the model, and converting the predicted tokens into secondary structure
labels.

.. code:: ipython3

    from sklearn import metrics
    
    def zero_shot_secondary_structure_prediction(model, sequence):
        model.eval()
        inputs = tokenizer(
            sequence, return_tensors="pt", padding="max_length", truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]  # Skip special tokens
        structure = [
            model.config.id2label[prediction.item()] for prediction in predictions[0]
        ]
        return "".join(structure)

4. **Predicting RNA Secondary Structure**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s now use the function we defined to predict the secondary structure
of an example RNA sequence. The sequence we’re using is:

``GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC``

.. code:: ipython3

    # Example RNA sequence
    sequence = "GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC"
    
    # Predict the secondary structure
    structure = zero_shot_secondary_structure_prediction(ssp_model, sequence)
    
    # The predicted structure should look something like this:
    print("Predicted structure:", structure)
    # Expected output: ..........((((....))))((((....))))((((...))))

5. **Using a Simplified Prediction API**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``OmniGenome`` model also provides a simpler interface for
predicting the secondary structure. You can directly use the ``fold``
method of the model to predict the structure in one line.

This method is especially useful when you want to avoid handling
tokenization and decoding manually.

.. code:: ipython3

    # Use the simplified fold method for prediction
    structure = ssp_model.fold(sequence)
    print("Predicted structure with fold method:", structure)
    # Expected output: ['..........((((....))))((((....))))((((...))))']

6. **Comparing with ViennaRNA**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For comparison, you can also use the popular RNA secondary structure
prediction tool ``ViennaRNA``. Below is an example of how to predict the
structure of the same sequence using ``ViennaRNA``.

You can install ``ViennaRNA`` by running:

.. code:: bash

   pip install viennarna

Then, use the following code to predict the structure.

.. code:: ipython3

    # Uncomment the following lines to use ViennaRNA
    # import ViennaRNA
    # print("ViennaRNA prediction:", ViennaRNA.fold(sequence)[0])
    # Expected output: ..........((((....))))((((....))))((((...))))

7. **Conclusion**
~~~~~~~~~~~~~~~~~

In this tutorial, we demonstrated how to use the ``OmniGenome`` model
for zero-shot RNA secondary structure prediction. We compared the
results with ``ViennaRNA`` and also showed how to use the simpler
``fold`` method for quick predictions.

The flexibility of ``OmniGenome`` allows for quick and efficient
secondary structure prediction for any RNA sequence, making it a
powerful tool in RNA research.
