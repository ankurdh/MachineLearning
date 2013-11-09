package edu.uncc.ml.naivetree;

import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instances;

public class NaiveTreeClassifier extends ClassifierTree {

	private static final long serialVersionUID = 1L;

	public NaiveTreeClassifier() {
		super(null);
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		super.buildClassifier(data);
		
		/**
		 * Following the guidelines to building a custom classifier: 
		 * http://weka.wikispaces.com/file/view/Build_classifier_353.pdf/82916711/Build_classifier_353.pdf
		 */
		data = new Instances(data);
		data.deleteWithMissingClass();
		
		/**
		 * This method initializes the attributes. Populates the 
		 */
		initializeAttributes(data);
		
		buildTree(data, true);
		
	}
	
	private void initializeAttributes(Instances instances){
		//TODO implement this.
	}
	
	@Override
	public void buildTree(Instances data, boolean arg1) throws Exception {
		super.buildTree(data, arg1);
	}
	
}
