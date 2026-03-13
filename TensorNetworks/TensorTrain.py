# Small experiment script for implementing tensor trains.
# This has not been subjected to rigorous validation
# and does not claim to be the most efficient possible
# implementation.

import numpy as np

class TensorTrain:
    def __init__(self, cores=[]):
        # assert cores[0].shape[0] == 1
        # assert cores[-1].shape[-1] == 1
        self.cores = cores

    def _get_elem(self, indices, cores, res):
        assert len(indices) == len(cores)
        if len(cores) == 1:
            return res @ cores[0][:, indices[0], :]
        else:
            new_res = res @ cores[0][:, indices[0], :]
            return self._get_elem(indices[1:], cores[1:], new_res)

    def get_elem(self, indices):
        return self._get_elem(indices[1:], self.cores[1:], \
                              self.cores[0][:, indices[0], :])

    def tt_svd(self, A, eps):
        # implementation from Oseledets (2011)
        delta = eps / np.sqrt(A.ndim - 1) * np.linalg.norm(A)
        r = [1]
        n = A.shape
        C = A
        for k in range(0, A.ndim - 1):
            C = C.reshape(r[k] * n[k], C.size // (r[k] * n[k]))
            U, S, Vh = np.linalg.svd(C)
            for s in range(S.size - 1, 0, -1):
                if np.linalg.norm(S[s:]) > delta:
                    break
            r.append(s + 1)
            U = U[:, :r[k + 1]]
            S = S[:r[k + 1]]
            Vh = Vh[:r[k + 1], :]
            self.cores.append(U.reshape(r[k], n[k], r[k + 1]))
            C = np.diag(S) @ Vh
        self.cores.append(C.reshape(2,3,1))


def test1():
    G0 = np.arange(4).reshape((1,2,2), order = 'F')
    G1 = np.arange(8).reshape((2,2,2), order = 'F')
    G2 = np.arange(4).reshape((2,2,1), order = 'F')
    return TensorTrain([G0, G1, G2], 3)
