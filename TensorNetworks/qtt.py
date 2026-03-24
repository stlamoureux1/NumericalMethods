# Naive implementation of Quantics Tensor Train using SVD

import numpy as np

class QTT:
    def __init__(self, f, tol=1e-10):
        self.cores = self.qtt_svd(f, tol)

    # Google Gemini
    def qtt_svd(self, f, tol=1e-10):
        """
        Decompose a 1D vector into a Quantic Tensor Train.
        f: input vector of size 2^L
        tol: relative accuracy tolerance
        """
        L = int(np.log2(len(f)))
        res = f.copy()
        cores = []
        r_prev = 1

        for k in range(1, L):
            # 1. Reshape the remainder to (r_prev * current_bit, all_future_bits)
            # Current bit is always 2 for Quantic
            matrix = res.reshape(r_prev * 2, -1)
            
            # 2. Perform SVD
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            
            # 3. Determine truncation rank based on tolerance
            # We find the smallest r such that the tail energy is < tol
            if tol > 0:
                # Calculate cumulative "energy" from smallest to largest
                error_sq = np.cumsum(s[::-1]**2)[::-1]
                # Threshold relative to total norm
                r_k = np.count_nonzero(error_sq > (tol * np.linalg.norm(s))**2)
                r_k = max(1, r_k) # Keep at least rank 1
            else:
                r_k = len(s)

            # 4. Truncate
            u_approx = u[:, :r_k]
            s_approx = s[:r_k]
            vh_approx = vh[:r_k, :]
            
            # 5. Store the core: Shape (r_prev, 2, r_k)
            cores.append(u_approx.reshape(r_prev, 2, r_k))
            
            # 6. Prepare the "remainder" for the next bit
            res = np.diag(s_approx) @ vh_approx
            r_prev = r_k

        # Final Core: The last remainder is just (r_prev, 2)
        # We reshape to (r_prev, 2, 1) to keep the 3-index format
        cores.append(res.reshape(r_prev, 2, 1))
        
        return cores

    # Google Gemini
    def reconstruct(self):
        # Start with the first core
        res = self.cores[0] # (1, 2, r1)
        for i in range(1, len(self.cores)):
            # This is a tensor contraction: 
            # (current_points, r_left) @ (r_left, 2, r_right)
            # We flatten the existing result to multiply by the next matrix
            r_left = self.cores[i].shape[0]
            res = res.reshape(-1, r_left) @ self.cores[i].reshape(r_left, -1)
        return res.flatten()

