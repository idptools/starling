Sequence Embeddings
==================

STARLING can generate **ensemble-aware sequence embeddings** that capture ensemble properties of intrinsically disordered proteins.
STARLING's sequence encoder was trained jointly with the diffusion model to produce embeddings that are informative for ensemble generation.

Basic Usage
-----------

You can generate sequence embeddings programmatically:

.. code-block:: python

    from starling import sequence_encoder

    sequence = "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK"
    embedding = sequence_encoder(sequence, aggregate=False)

The ``aggregate`` parameter controls whether to return per-residue embeddings or a single aggregated embedding (mean-pooled) for the entire sequence. 
Single aggregated embeddings are useful when comparing sequences of the same length, while per-residue embeddings are useful for downstream tasks that require residue-level information.

Advanced Usage
-------------

Processing Multiple Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sequence encoder accepts various input formats including single sequences, lists, dictionaries, and FASTA files:

.. code-block:: python

    # Process a list of sequences
    sequences = [
        "GSGSGSGSGSGS",
        "ACDEFGHIKLMNPQRSTVWY"
    ]
    embeddings = sequence_encoder(sequences, salt=150)
    
    # Process sequences from a dictionary
    seq_dict = {
        "protein_A": "GSGSGSGSGSGS",
        "protein_B": "ACDEFGHIKLMNPQRSTVWY"
    }
    embeddings = sequence_encoder(seq_dict, salt=150)
    
    # Process sequences from a FASTA file
    embeddings = sequence_encoder("path/to/sequences.fasta", salt=150)

Controlling Ionic Strength
~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING's encoder was trained at different ionic strengths. You can specify the ionic strength to model specific conditions:

.. code-block:: python

    # Generate embeddings at physiological ionic strength (150mM)
    physiological = sequence_encoder(sequence, salt=150)
    
    # Generate embeddings at low ionic strength (20mM)
    low_salt = sequence_encoder(sequence, salt=20)
    
    # Generate embeddings at high ionic strength (300mM)
    high_salt = sequence_encoder(sequence, salt=300)

Output Options
~~~~~~~~~~~~

Control how embeddings are returned and saved:

.. code-block:: python

    # Return per-residue embeddings (default)
    per_residue = sequence_encoder(sequence, salt=150, aggregate=False)
    
    # Return a single embedding vector per sequence
    aggregated = sequence_encoder(sequence, salt=150, aggregate=True)
    
    # Save embeddings to disk
    sequence_encoder(
        sequence,
        salt=150,
        output_directory="results/embeddings"
    )

