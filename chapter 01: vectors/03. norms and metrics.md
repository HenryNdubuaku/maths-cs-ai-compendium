# Metrics and Norms

- We know vectors have magnitude and direction. But how do we actually measure "how big" a single vector is, or "how far apart" two vectors are? This is where **norms** and **metrics** come in.

- In scalars, we know that 10 > 5, because their values quantify them, but how can we quantify a vector? It's **norm**, it measures the size of a single vector. 

- The most familiar norm is the **Euclidean norm** (L2), which is just the magnitude formula we already know:

$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

- But there are other ways to measure size. Imagine you are in a city with a grid of streets. You cannot walk diagonally through buildings, so the "length" of your journey is the total blocks walked along each street. This is the **Manhattan norm** (L1):

$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

- Or you might only care about the single largest component, ignoring the rest. This is the **Max norm** (L-infinity):

$$\|\mathbf{v}\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

- All three are special cases of the **general Lp norm**:

$$\|\mathbf{v}\|_p = (|v_1|^p + |v_2|^p + \cdots + |v_n|^p)^{1/p}$$

- Setting $p = 2$ gives Euclidean, $p = 1$ gives Manhattan, and as $p \to \infty$ you get the Max norm. As $p$ grows, the largest component contributes more and more, until eventually only it matters.

- Every norm must obey three rules:

    - **Non-negativity**: $\|\mathbf{v}\| \geq 0$, and $\|\mathbf{v}\| = 0$ only if $\mathbf{v} = \mathbf{0}$. Size is never negative, and only the zero vector has zero size.

    - **Scaling**: $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$. Doubling a vector doubles its size.

    - **Triangle inequality**: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$. The shortcut is never longer than going the long way round.

- Now, a **metric** measures the distance *between* two vectors. Think of it as asking: "how far apart are these two points?"

- The simplest way to get a metric is to use a norm on the difference: $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$. Subtract the two vectors, then measure the size of what remains.

- Using the Euclidean norm this gives us the familiar **Euclidean distance**:

$$d(\mathbf{u}, \mathbf{v}) = \sqrt{(u_1 - v_1)^2 + (u_2 - v_2)^2 + \cdots + (u_n - v_n)^2}$$

- Using the Manhattan norm gives **Manhattan distance**, the total difference along each axis, like counting city blocks between two locations.

- Every metric must obey four rules:

    - **Non-negativity**: $d(\mathbf{u}, \mathbf{v}) \geq 0$. Distance is never negative.

    - **Identity**: $d(\mathbf{u}, \mathbf{v}) = 0$ if and only if $\mathbf{u} = \mathbf{v}$. Zero distance means the same point.

    - **Symmetry**: $d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$. The distance from A to B is the same as from B to A.

    - **Triangle inequality**: $d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$. Going directly is never longer than taking a detour.

- So what is the relationship between the two? A norm measures one vector, a metric measures the gap between two. Every norm naturally creates a metric (by measuring the difference), but not every metric comes from a norm.

- For example, **Hamming distance** counts the number of positions where two vectors differ. It is a valid metric, but it does not come from any norm.

- In ML, choosing the right norm or metric matters. 

- L2 distance squares each difference before summing, so a single large difference dominates the result.

- L1 distance sums the absolute differences, treating each one equally. A single large difference has less influence compared to L2.

## Coding Tasks (use CoLab or notebook)

1. Compute L1, and L2 norms of the same vector. Try changing the values and notice which norm is most sensitive to large components vs many small ones. Then try computing the Lp norm for increasing values of p (e.g. 1, 2, 5, 10, 50, 100) and watch it converge towards the L-infinity value.
```python
import jax.numpy as jnp

v = jnp.array([3.0, -4.0, 1.0])

l1 = jnp.sum(jnp.abs(v))
l2 = jnp.sqrt(jnp.sum(v ** 2))

print(f"L1: {l1}, L2: {l2:.2f}")
```

2. Compute the Euclidean and Manhattan distance between two vectors. Try moving the vectors closer or further apart and observe how each distance responds differently.
```python
import jax.numpy as jnp

u = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([4.0, 0.0, 1.0])

euclidean = jnp.sqrt(jnp.sum((u - v) ** 2))
manhattan = jnp.sum(jnp.abs(u - v))

print(f"Euclidean: {euclidean:.2f}, Manhattan: {manhattan}")
```
