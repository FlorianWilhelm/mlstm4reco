# -*- coding: utf-8 -*-
"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
from torch import nn

from spotlight.layers import ScaledEmbedding, ZeroEmbedding

from .layers import mLSTM

PADDING_IDX = 0


class mLSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [2] Hidasi, Balazs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(mLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        h_init = torch.zeros(embedding_dim)
        h_init.normal_(0, 1.0 / self.embedding_dim)
        self.h_init = nn.Parameter(h_init, requires_grad=True)

        self.mlstm = mLSTM(input_size=embedding_dim,
                           hidden_size=embedding_dim)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """
        # Make the embedding dimension the channel dimension
        sequence_embeddings = self.item_embeddings(item_sequences)

        # pad from left with initial state
        batch_size = sequence_embeddings.size()[0]
        embedding_dim = self.h_init.size()[0]
        seq_start = self.h_init.expand(batch_size, embedding_dim)

        user_representations = self.mlstm(sequence_embeddings, (seq_start, seq_start))

        user_representations = user_representations.permute(0, 2, 1)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot
