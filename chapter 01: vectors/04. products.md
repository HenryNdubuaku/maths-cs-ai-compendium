# Vector Products

- We have seen how to add and scale vectors. But can we *multiply* two vectors together? It turns out there is more than one way to do it, and each answers a different question.

- An **inner product** is the general idea: a function that takes two vectors and produces a single number (a scalar). It is the abstract blueprint for "multiplying" vectors.

- Any inner product must satisfy three rules:

    - **Positive definiteness**: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$, and equals zero only for the zero vector. Multiplying a vector with itself always gives a non-negative result.

    - **Symmetry**: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$. The order does not matter.

    - **Linearity**: $\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$. It distributes over addition and scaling.

- The **dot product** is the most common inner product. It is the concrete version you will use almost everywhere. For two vectors $\mathbf{a} = (a_1, a_2, \ldots, a_n)$ and $\mathbf{b} = (b_1, b_2, \ldots, b_n)$:

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

- Multiply matching components, then add everything up. That is all it is.

- But what does this number *mean*? The dot product has a beautiful geometric interpretation:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos(\theta)$$

![Dot product: vector a projected onto b, with angle θ and projection shown](../images/dot_product.svg)

- This connects the dot product directly to the angle $\theta$ between the two vectors. The result tells you how much the two vectors "agree" in direction.

- If they point the same way ($\theta = 0°$), $\cos(\theta) = 1$ and the dot product is maximised.

- If they are orthogonal ($\theta = 90°$), $\cos(\theta) = 0$ and the dot product is exactly zero. This gives us a precise test for orthogonality.

- If they point in opposite directions ($\theta = 180°$), $\cos(\theta) = -1$ and the dot product is negative.

- A vector dotted with itself gives its magnitude squared: $\mathbf{a} \cdot \mathbf{a} = \|\mathbf{a}\|^2$.

- The dot product also gives us **projection**, the shadow one vector casts onto another. The projection of $\mathbf{a}$ onto $\mathbf{b}$ is:

$$\text{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \, \mathbf{b}$$

- Think of shining a light straight down onto $\mathbf{b}$. The shadow of $\mathbf{a}$ on that line is the projection. It tells you how much of $\mathbf{a}$ lies in the direction of $\mathbf{b}$.

- **Cosine similarity** normalises the dot product by dividing out both magnitudes:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

- This gives a value between $-1$ and $1$ that measures direction alignment, ignoring how long the vectors are. It is widely used in ML to compare things like documents, embeddings, and user preferences.

- Now, the dot product takes two vectors and returns a scalar. The **cross product** does the opposite, it takes two vectors and returns a *new vector*.

- The cross product $\mathbf{a} \times \mathbf{b}$ produces a vector that is perpendicular to both $\mathbf{a}$ and $\mathbf{b}$:

$$\mathbf{a} \times \mathbf{b} = (a_2 b_3 - a_3 b_2, \; a_3 b_1 - a_1 b_3, \; a_1 b_2 - a_2 b_1)$$

- The cross product only works in 3D. While the dot product works in any number of dimensions, the cross product is specific to three-dimensional space.

- Its magnitude equals the area of the parallelogram formed by the two vectors:

$$\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\| \, \|\mathbf{b}\| \sin(\theta)$$

- Notice the pattern: the dot product uses $\cos(\theta)$ and the cross product uses $\sin(\theta)$. The dot product measures how much two vectors align, the cross product measures how much they *differ* in direction.

- The direction of the result follows the **right-hand rule**: curl the fingers of your right hand from $\mathbf{a}$ towards $\mathbf{b}$, and your thumb points in the direction of $\mathbf{a} \times \mathbf{b}$.

- Unlike the dot product, the cross product is **not commutative**: $\mathbf{a} \times \mathbf{b} = -(\mathbf{b} \times \mathbf{a})$. Swapping the order flips the direction.

- If two vectors are parallel, their cross product is the zero vector (since $\sin(0°) = 0$). No area, no perpendicular direction.

- What happens when you combine three vectors using both products? This gives us **triple products**.

- The **scalar triple product** $\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$ first takes the cross product of two vectors, then dots the result with the third. The output is a single number that equals the volume of the parallelepiped (a slanted 3D box) formed by the three vectors.

- If the scalar triple product is zero, the three vectors are **coplanar**, they all lie in the same flat plane and form no volume.

- The order can be cycled without changing the result: $\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = \mathbf{b} \cdot (\mathbf{c} \times \mathbf{a}) = \mathbf{c} \cdot (\mathbf{a} \times \mathbf{b})$.

- The **vector triple product** $\mathbf{a} \times (\mathbf{b} \times \mathbf{c})$ applies the cross product twice and returns a vector. It expands neatly using the identity:

$$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = (\mathbf{a} \cdot \mathbf{c})\mathbf{b} - (\mathbf{a} \cdot \mathbf{b})\mathbf{c}$$

- The result always lies in the plane spanned by $\mathbf{b}$ and $\mathbf{c}$. Note that the cross product is **not associative**: $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) \neq (\mathbf{a} \times \mathbf{b}) \times \mathbf{c}$.

## Coding Tasks (use CoLab or notebook)

1. Compute the dot product of two vectors and use it to find the angle between them. Try making them orthogonal, parallel, or opposite and see how the angle changes.
```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, -1.0, 2.0])

dot = jnp.dot(a, b)
angle = jnp.arccos(dot / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))

print(f"Dot product: {dot}")
print(f"Angle: {jnp.degrees(angle):.1f}°")
```

2. Compute the cross product of two 3D vectors and verify the result is perpendicular to both by checking that its dot product with each original vector is zero.
```python
import jax.numpy as jnp

a = jnp.array([1.0, 0.0, 0.0])
b = jnp.array([0.0, 1.0, 0.0])

cross = jnp.cross(a, b)

print(f"a x b = {cross}")
print(f"Perpendicular to a: {jnp.dot(cross, a) == 0}")
print(f"Perpendicular to b: {jnp.dot(cross, b) == 0}")
```
