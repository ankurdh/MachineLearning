package edu.uncc.ml.naivetree;

import weka.classifiers.trees.j48.ClassifierTree;

public class TreePruner {

	/**
	 *
	 * The pruning algorithm must prune based on the paper: http://ai.stanford.edu/~ronnyk/treesHB.pdf
	 * Also, Weka Framework provides a Class named PrunableClassifierTree that has a pruning implmentation.
	 * Its possible to make the NaiveDecisionTree inherit from the PrunableClassifierTree class and invoke the
	 * prune() method of the PrunableClassifierTree class
	 * @param tree The tree that has to be pruned.
	 * @throws Exception
	 */

	public static void pruneTree(ClassifierTree tree) throws Exception {
		throw new Exception("Tree Pruning Not Implemented Yet.");
	}

}
