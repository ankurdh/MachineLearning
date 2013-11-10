package edu.uncc.ml.naivetree;

import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveTree extends Classifier {

	private static final long serialVersionUID = 4097340583224063859L;
	
	private ClassifierTree naiveTree = null;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		naiveTree = new NaiveTreeClassifier();
		naiveTree.buildClassifier(instances);
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
	    return naiveTree.classifyInstance(instance);
	}
}
