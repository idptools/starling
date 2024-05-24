# import key modules
import numpy as np
import torch
from finches.epsilon_calculation import Interaction_Matrix_Constructor
from finches.forcefields.mPiPi import mPiPi_model


def epsilon_vector(sequence):
    # Initialize a finches.forcefields.Mpipi.Mpipi_model object
    Mpipi_GGv1_params = mPiPi_model(version="mPiPi_GGv1")

    # initialize an InteractionMatrixConstructor
    IMC = Interaction_Matrix_Constructor(parameters=Mpipi_GGv1_params)

    attractive, repulsive = IMC.calculate_epsilon_vectors(sequence, sequence)
    # Pad with zeros up to length 384
    attractive_padded = np.pad(attractive, (0, 384 - len(attractive)), "constant")
    repulsive_padded = np.pad(repulsive, (0, 384 - len(repulsive)), "constant")

    # Concatenate the padded arrays
    epsilon_vector = np.concatenate((attractive_padded, repulsive_padded))

    return epsilon_vector.tolist()
