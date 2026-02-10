# Vector Spaces

- Think of a Vector Space as a specific kind of playground where mathematical objects live.

- Each object is called a **vector**. 

- The vectors in the vector space live under a very specific, unbreakable set of rules:

    - **Vector Addition (Combining)**:
    You can take any two vectors and combine them to create a new one.
    Think of vectors as instructions for movement.
    If vector A means "walk 3 steps forward" and vector B means "walk 2 steps right,"
    adding them (A + B) creates a new, single instruction: "walk 3 steps forward and 2 steps right."

    - **Scalar Multiplication (Scaling)**:
    You can take any vector and scale it using a regular number (a "scalar").
    You can stretch it, shrink it, or reverse it.
    If vector A is "walk 3 steps forward," scaling it by 2 makes it "walk 6 steps forward."
    Scaling it by -1 flips it entirely to "walk 3 steps backward."

- For geometric intuition in machine learning (ML), we will always think of vectors as a point in Euclidean space, represented by it's coordinates. 

- The vector $\mathbf{a}$ (denoted mathematically as lowercase letters in bold) has $n$ coordinates, each representing a position along an axis.

$$\mathbf{a} = [a_1, a_2, a_3]$$

![A vector a = (3, 2, 4) plotted in 3D space with x, y, z axes](../images/vector_3d.svg)

- The **dimension** of a vector space is the number of independent directions it contains. $\mathbb{R}^2$ is 2-dimensional (needs 2 coordinates), while $\mathbf{a}$ above lives in $\mathbb{R}^3$.

- We can for instance represent any object, say, a human, as a vector, where $h_1$ = height in cm, $h_2$ = weight in kg, $h_3$ = age, $h_4$ = waist size in cm.

$$\mathbf{h} = [185, 75, 30, 44]$$

- We have now created a vector space with a vector representing a human, and we can represent multiple humans. 

- Notice how this becomes hard to visualise at dimension = 4. Well in in ML, it gets a lot bigger...a lot!

- For instance, we can add more features, creating a rich representation of a human, often called feature vectors in ML.

- The more unique and meaningful features you have, the more descriptive the feature vector is, an important factor to remember. 

- Now, **Linear algebra** is the study of vectors, vector spaces and mappings between vectors.

- We represent almost every thing in AI/ML as vectors, making linear algebra the bedrock of the field.

- Vector addition can be performed by placing one vector on the tail of the other visually, and drawing from the origin to the endpoint.

![Vector addition: a (red) plus b (blue) gives resultant a + b (green dashed)](../images/vector_addition.svg)

- For two vectors $\mathbf{a} = (a_1, a_2)$ and $\mathbf{b} = (b_1, b_2)$: $\mathbf{a} + \mathbf{b} = (a_1 + b_1, a_2 + b_2)$

- Vectors can also be subtracted, with all addition rules applying too.

- Multiplying a vector by a scalar scales the vector by that factor in the same direction.

![Scalar multiplication: v (red), 2v (blue, doubled), -v (purple, reversed)](../images/scalar_multiplication.svg)

- For a scalar $c$ and vector $\mathbf{v} = (v_1, v_2)$: $c\mathbf{v} = (cv_1, cv_2)$

- **Closure under Addition**: If you add any two vectors from the vector space, the result is also a vector within the same space: If $\mathbf{u} \in V$ and $\mathbf{v} \in V$, then $\mathbf{u} + \mathbf{v} \in V$

- **Closure under Scalar Multiplication**: If you multiply any vector from the vector space by a scalar, the result is a vector within the same space: If $\mathbf{v} \in V$ and $c \in F$, then $c\mathbf{v} \in V$

- **Associativity of Addition**: For any three vectors $\mathbf{u}$, $\mathbf{v}$, and $\mathbf{w}$: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$

- **Commutativity of Addition**: For any two vectors $\mathbf{u}$ and $\mathbf{v}$: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$

![Parallelogram rule: both paths (u then v, or v then u) reach the same point](../images/commutativity.svg)

- Both paths through the parallelogram arrive at the same point.

- **(Zero Vector)**: There exists a vector $\mathbf{0}$ such that for any vector $\mathbf{v}$: $\mathbf{v} + \mathbf{0} = \mathbf{v}$

![Zero vector: v + 0 = v](../images/zero_vector.svg)

- **Additive Inverse**: For every vector $\mathbf{v}$, there exists a vector $-\mathbf{v}$ such that: $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

![Additive inverse: v (red) and -v (blue) cancel to zero](../images/additive_inverse.svg)

- **Distributivity 1**: For any scalar $c$ and vectors $\mathbf{u}$, $\mathbf{v}$: $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$

![Distributivity: scaling the sum (gold) equals summing the scaled vectors](../images/distributivity.svg)

- Scaling the sum (gold) gives the same result as summing the scaled vectors.

- **Distributivity 2**: For any scalars $c$, $d$ and vector $\mathbf{v}$: $(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$

- **Associativity**: For any scalars $c$, $d$ and vector $\mathbf{v}$: $(cd)\mathbf{v} = c(d\mathbf{v})$

- **Identity Element**: For any vector $\mathbf{v}$: $1\mathbf{v} = \mathbf{v}$, where $1$ is the multiplicative identity in the field of scalars.

- **What is NOT a vector space**: The set of all vectors in $\mathbb{R}^2$ where $x_1 > 0$ is not a vector space. It fails closure: scaling $\mathbf{v} = (3, 1)$ by $-1$ gives $(-3, -1)$, which has $x_1 < 0$ and leaves the set. A valid vector space must survive every addition and every scaling without exception.
