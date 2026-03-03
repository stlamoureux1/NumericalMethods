## Background

The main issue with low-rank approximation for dynamical problems is preventing the rank of the solution from growing as we advance in time. We assume that our solution lives in some tensor product space of Hilbert spaces. In general, though, the elements of a certain fixed rank inside this space *do not form a linear subspace* but rather a manifold. Luckily for us, an equation with a first derivative in time tells us just what we need to know about the *tangent space* to that manifold. We can keep our rank under control if we use this information to project onto the tangent space at each time step. 

In order to this, we need to find the orthogonal projection onto the tangent space, which is the same as finding an approximation $\tilde{u}$ such that $\langle \dot{u} - \dot{\tilde{u}}, v \rangle$ for all $v$ in the tangent space. 

## Derivation

In order to calculate the low-rank approximation, we begin with the rank $r$ ansatz

$$
    u(t, x, y) \approx \tilde{u}(t, x, y) = \sum_{ij}^{r}X_i(t,x)S_{ij}(t)Y_j(t,y).
$$

In practice, this is usually just the SVD of the discretized version of the problem, as shown below.
To advance the solution in time, we take discrete samples at each time step $n$

$$
    \tilde{u}^n(x, y) = \sum_{ij}X_i^n(x)S_{ij}^n Y_j^n(y).
$$


### Projector-splitting integrators
Taking our ansatz, finding the time derivative, and solving for each component function is a straightforward exercise (see Koch & Lubich). But, doing so involves taking the inverse of our singular value matrix $S$, which becomes a problem for very small singular values. In particular, if we *overestimate* the rank of our solution, we're in trouble.

The alternative offered by Lubich & Oseledets is to adopt a *splitting* approach. Let $F(u, t, x, y)$ be the right-hand side of dynamic equation $\partial_t u = F(u, t, x, y)$. The method proceeds as follows:

1. Find the low-rank approximation of the initial condition:

$$
\tilde{u}(0,x,y) = \sum_{ij} X_i(0,x) S_{ij}(0) Y_j(0,y).
$$

2. For each time step $t$ do:
   ##### K-step
   - let $K_j^t = \sum_{i} X_i(t, x) S_{ij}(t)$
  
   - solve the ODE: $\dot{K}_j^t = \bigg\langle Y_j^t, F(\sum_k K_k^t Y_k^t) \bigg\rangle $
  
   - Use the solution to get $K_j^{t+1} = K_j(t+1,x)$
  
   - find $X_i^{t+1}, \hat{S}_{ij}$ by orthonormal decomposition of the resulting $K_j^{t+1}$

   ##### S-step
   - let $S_{ij}^t = \hat{S}_{ij}$

   - solve: $\dot{S}_{ij} = - \bigg\langle X_i^{t+1}Y_i^{t}, F(\sum_{kl} X_{k}^{t+1} S_{kl} Y_{l}^t) \bigg\rangle$
  
   - Use the solution to get $\tilde{S}_{ij} = S_{ij}(t+1)$

   ##### L-step
   - let $L_i^t = \sum_{j} \tilde{S}_{ij} Y_j^t$

   - solve: $\dot{L}_i = \bigg\langle X_i^{t+1}, F(\sum_{l} X_{l}^{t+1} L_{l}^t) \bigg\rangle$
  
   - Use the solution to get $L_i^{t+1} = L_i(t+1,y)$
  
   - find $S_{ij}^{t+1}, Y_{j}^{t+1}$ by orthonormal decomposition of the resulting $L_i^{t+1}$

4. The result for each time step is $\tilde{u}^t = \sum_{ij} X_i^t S_{ij}^t Y_{j}^{t}$
