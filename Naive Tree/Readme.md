Project Outline
----------------------------------------------------------------
Decision trees are a widely used data structure. This has brought a keen interest in scaling up the decision tree algorithm for faster construction. This algorithm targets the datasets in which the "Attribute Independence" assumption can be made. With this assumption the overall complexity of the construction of the decision tree can be brought down to O(mn) compared to O(mn2) of C4.5. However, this will require an additional space linear in 'n', where n-> no. of attributes and m-> number of training examples.
----------------------------------------------------------------
Approach
----------------------------------------------------------------
Consider the Information Gain formula:
IG(S, X) = Entropy(S) - Σ (|Sx|/|S|) Entropy (Sx),
Entropy(S) = -ΣPs(Ci) . log(Ps(Ci))

The most time consuming part of the above process is the calculation of Entropy(Sx) [Sx is the subset of the dataset 'S' which have the attribute X = x, Ci is the ith class]. This calculation will requires finding the subset of the original set 'S' which satisfy all the attribute conditions which are in the parent nodes of the node is consideration. This requires passing through all the attributes of 'S' to estimate P(Ci | Xp, x):

	P(Ci | Xp, X) = P(Ci | X)P(X| Xp, Ci)/SUM_OVER_ALL_CLASSES(P(Ci| Xp)P(X|Xp, Ci))

which is the probability of the class 'Ci' given all the conditions of the parent nodes(Xp) and the attribute value of current node: 'x' hold. But if the attribute at the node in consideration is independent of each of the attribute in all its parents till the root, the probability term P(X | Xp, C) is the same as P(X | C), implying:

	P(Ci | Xp, X) = P(Ci | X)P(X| Ci)/SUM_OVER_ALL_CLASSES(P(Ci| Xp)P(X|Ci))

where, Xp is the set of attribute conditions in the parents, X is the current attribute and C is the set of classes. Now, the term P(X | C) can be pre-computed before the algorithm of construction begins and saved for instant access when required. This is known as the Independent Information Gain{IIG} and is the core idea of the fast construction. C4.5 takes up a time of O(n) in this operation that makes its overall complexity O(mn2). With this assumption, the "Naive Tree" Algorithm based on the conditional independence assumption is as follows:
------------------------------------------------------------------------------------------
Algorithm NT(π,S)
Input : π is a set of candidate attributes, and
S is a set of labeled instances
Output : A decision tree T.
1. Compute PS(Ci) on S for each class Ci.
2. For each attribute X in π , compute
IIG ( S , X ) based on Entropy Equation and Equation (1)
3. Use the attribute X max with the highest IIG for the root.
4 Partition S into disjoint subsets Sx using X max .
5. For all values x of X max
T x = NT ( π - Xmax , Sx)
Add T x as a child of X max
6. Return T
