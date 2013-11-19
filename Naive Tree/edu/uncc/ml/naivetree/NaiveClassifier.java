package edu.uncc.ml.naivetree;

import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveClassifier extends Classifier implements Drawable {

	private static final long serialVersionUID = 4097340583224063859L;
	private ClassifierTree naiveTree = null;

	/**
	 * The Weka classifier requiers overriding of several other methods. Right now, 
	 * overriding only the required ones. 
	 */
	
	@Override
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		
		sb.append("-----------------------------------------------\n");
		sb.append("Naive Tree\nImplemented from paper: http://www.cs.unb.ca/profs/hzhang/publications/AAAI06.pdf\n" +
				"Author: Ankur Huralikoppi\n");
		sb.append("-----------------------------------------------\n");
		
		return sb.toString();
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		naiveTree = new NaiveDecisionTree();
		naiveTree.buildClassifier(instances);
		String graph = naiveTree.graph();
		System.out.println(graph);
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
	    return naiveTree.classifyInstance(instance);
	}

	@Override
	public String graph() throws Exception {
		return naiveTree.graph();
	}

	@Override
	public int graphType() {
		return Drawable.TREE;
	}
}
