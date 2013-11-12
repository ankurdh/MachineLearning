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
		 * Override and invoke the build tree method of the ClassifierTree class
		 */
		buildTree(data, true);
		
	}
	
	private void initializeAttributes(Instances instances){
		attributes = AttributeHelper.initializeAttributes(instances, attributes);
	}
	
	/**
	 * Overriding the core buildTree method of the WEKA framerowk.
	 */
	@Override
	public void buildTree(Instances data, boolean arg1) throws Exception {
		
		m_localModel = new NoSplit(new Distribution(data));
		
		/**
		 * This method initializes the attributes. Populates the required tables before the tree construction according to the paper. 
		 */
		initializeAttributes(data);
		
		/**
		 * m_localModel is the variable which has to be set. This is the one which will have the split. 
		 */
		int splitAttributeIndex = -1;
		
		while(true){

			splitAttributeIndex = getAttributeIndexWithMaxIIGFromData(data);
			if(splitAttributeIndex == -1)
				break;
			
			System.out.println("Best IIG Attribute: " + splitAttributeIndex);
			
			C45Split currentNodeSplit = new C45Split(splitAttributeIndex, 2, new Distribution(data).total());
			currentNodeSplit.buildClassifier(data);

			//remove the current attribute for the children. so that they don't consider it.
			if(data.attribute(splitAttributeIndex).isNominal())
				attributes[splitAttributeIndex] = new DiscreteDataAttributeImpl(data, splitAttributeIndex);
			else if(data.attribute(splitAttributeIndex).isNumeric())
				attributes[splitAttributeIndex] = new ContinuousDataAttributeImpl(data, splitAttributeIndex);
			
			if(currentNodeSplit.numSubsets() > 1){
				m_localModel = currentNodeSplit;
				break;
			}
			
			System.out.println("Best Att: " + splitAttributeIndex);
			
		}
		
		System.out.println("Current Node children: " + m_localModel.numSubsets());
		
		/**
		 * Build the subtrees now. If the currentNodeSplit.numSubsets == 1, then this is a leaf node
		 */
		if(m_localModel.numSubsets() < 1)
			m_isLeaf = true;
		else {
			
			//There are going to be #currentNodeSplit.numSubsets() subtrees for this node. Construct each of them.
			//Create data subsets for each split
			Instances [] subtreeSplitData = m_localModel.split(data);
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			
			for(int i = 0; i < m_localModel.numSubsets(); i ++){
				NaiveTreeClassifier newSubTree = new NaiveTreeClassifier();
				newSubTree.buildTree(subtreeSplitData[i++], true);
				
				m_sons[i] = newSubTree;
			}
			
			System.out.println("Constructed Subtrees.");
			
		}
		
		//TODO remove this once all the required implementation is completed
//		throw new Exception("Not implemented yet");
		
	}
	
	private int getAttributeIndexWithMaxIIGFromData(Instances data){

		/**
		 * Trigger attribute Independent Information Gain calculation here. 
		 * The data for the following call will recursively reduce. Update the attribute entropies with respect to new data everytime.
		 */
		AttributeHelper.calculateAttributeEntropies(data, attributes);
		
		/**
		 * Now we have attribute IIGs. Iterate over all attributes and return the one with maximum IIG.
		 */
		double currentMaxIIG = 0;
		int maxAttributeIndex = -1;
		
		for(int attributeIndex = 0 ; attributeIndex < data.numAttributes(); attributeIndex++){
			
			if(attributeIndex == data.classIndex())
				continue;
			
			/**
			 * Attributes might not have any data entries qualifying them. This bug occurred when the attribute impurity is 0.  
			 */
			
			if(attributes[attributeIndex].getAttributeIIG() == -1.0)
				continue;
			
			if(attributes[attributeIndex].getAttributeIIG() > currentMaxIIG){
				currentMaxIIG = attributes[attributeIndex].getAttributeIIG();
				maxAttributeIndex = attributeIndex;
			}
		}
		
		return maxAttributeIndex;
	}
	
}
