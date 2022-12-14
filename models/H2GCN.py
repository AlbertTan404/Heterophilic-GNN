"""
@author: Albert Tan
@create-date: 2022-12-14
@last-edit-date: 2022-12-14
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class H2GCN(nn.Layer):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            n_class: int,
            k: int = 2,
            dropout: float = 0.5
    ) -> None:
        super(H2GCN, self).__init__()
        self.k = k

        self.embed_layer = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features=hidden_dim * (2 ** (k + 1) - 1), out_features=n_class)

    def forward(
            self,
            x: paddle.Tensor,
            adj: paddle.sparse.sparse_csr_tensor,
            adj_2hop: paddle.sparse.sparse_csr_tensor
    ) -> paddle.Tensor:
        x = self.embed(x)

        hidden_reps = [x]
        for _ in range(self.round):
            r1 = adj.matmul(x)
            r2 = adj_2hop.matmul(x)
            x = paddle.concat([r1, r2], axis=-1)
            hidden_reps.append(x)

        hf = self.dropout(paddle.concat(hidden_reps, axis=-1))

        return F.log_softmax(self.classification(hf), axis=1)
