package edu.uncc.ml.naivetree;

import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AttributeHelper;
import edu.uncc.ml.naivetree.attributes.DataAttribute;
import edu.uncc.ml.naivetree.attributes.impl.ContinuousDataAttributeImpl;
import edu.uncc.ml.naivetree.attributes.impl.DiscreteDataAttributeImpl;

public class NaiveTreeClassifier extends ClassifierTree {

	private static final long serialVersionUID = 1L;

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
		
		/**
		 * Override and invoke the build tree method of the ClassifierTree class
		 */
		buildTree(data, true);
		
	}
	
	private void initializeAttributes(Instances instances){
		AttributeHelper.initializeAttributes(instances);
	}
	
	/**
	 * Overriding the core buildTree method of the WEKA framerowk.
	 */
	@Override
	public void buildTree(Instances data, boolean arg1) throws Exception {
		
		m_localModel = new NoSplit(new Distribution(data));
		
		/**
		 * Calculate the Attribute entropies and IIGs here. 
		 */
		
		AttributeHelper.calculateAttributeEntropies(data);
		
		/**
		 * m_localModel is the variable which has to be set. This is the one which will have the split. 
		 */
		int splitAttributeIndex = -1;
		boolean subTreeConstructionPossible = false;
		
		while(true){

			splitAttributeIndex = AttributeHelper.getAttributeIndexWithMaxIIGFromData(data);
			if(splitAttributeIndex == -1)
				break;
			
			System.out.println("\nBest IIG Attribute: " + splitAttributeIndex);
			
			C45Split currentNodeSplit = new C45Split(splitAttributeIndex, 2, new Distribution(data).total());
			currentNodeSplit.buildClassifier(data);

			/**
			 * remove the current attribute for the children. so that they don't consider it.
			 */
			if(data.attribute(splitAttributeIndex).isNominal()){
				DataAttribute newAttribute = new DiscreteDataAttributeImpl(data, splitAttributeIndex);
				newAttribute.setDeleted(true);
				AttributeHelper.setAttributeValue(splitAttributeIndex, newAttribute);
			}
			else if(data.attribute(splitAttributeIndex).isNumeric()) {
				DataAttribute newAttribute = new ContinuousDataAttributeImpl(data, splitAttributeIndex);
				newAttribute.setDeleted(true);
				AttributeHelper.setAttributeValue(splitAttributeIndex, newAttribute);
			}
			
			System.out.println("Current Node children: " + currentNodeSplit.numSubsets());
			
			if(currentNodeSplit.numSubsets() > 1){
				m_localModel = currentNodeSplit;
				subTreeConstructionPossible = true;
				break;
			}
		}
		
		/**
		 * Build the subtrees now. If the currentNodeSplit.numSubsets == 1, then this is a leaf node
		 */
		if(subTreeConstructionPossible) {
			
			//There are going to be #currentNodeSplit.numSubsets() subtrees for this node. Construct each of them.
			//Create data subsets for each split
			Instances [] subtreeSplitData = m_localModel.split(data);
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			
			for(int i = 0; i < m_localModel.numSubsets(); i ++){
				NaiveTreeClassifier newSubTree = new NaiveTreeClassifier();
				newSubTree.buildTree(subtreeSplitData[i], true);
				
				m_sons[i] = newSubTree;
			}
			
			System.out.println("Constructed Subtrees.");
			
		} else {
			//there has to be just one possible split. Mark this as the leaf.
			m_isLeaf = true;
		}
		
		System.out.println("\nRecursing Back..");
		
	}	
}
