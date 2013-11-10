package edu.uncc.ml.naivetree;

import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AttributeHelper;
import edu.uncc.ml.naivetree.attributes.DataAttribute;

public class NaiveTreeClassifier extends ClassifierTree {

	private static final long serialVersionUID = 1L;
	private DataAttribute [] attributes;

	public NaiveTreeClassifier() {
		super(null);
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		/**
		 * Following the guidelines to building a custom classifier: 
		 * http://weka.wikispaces.com/file/view/Build_classifier_353.pdf/82916711/Build_classifier_353.pdf
		 */
		data = new Instances(data);
		data.deleteWithMissingClass();
		
		/**
		 * This method initializes the attributes. Populates the required tables before the tree construction according to the paper. 
		 */
		initializeAttributes(data);
		
		buildTree(data, true);
		
	}
	
	private void initializeAttributes(Instances instances){
		attributes = AttributeHelper.initializeAttributes(instances, attributes);
	}
	
	@Override
	public void buildTree(Instances data, boolean arg1) throws Exception {
		throw new Exception("Not implemented yet");
	}
	
}
