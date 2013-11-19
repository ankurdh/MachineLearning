package edu.uncc.ml.naivetree;

import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AttributeHelper;

public class NaiveDecisionTree extends ClassifierTree {

	private static final long serialVersionUID = 1L;

	public NaiveDecisionTree() throws Exception {
		/**
		 * Necessary to invoke the constructor of the ClassifierTree class
		 * with an instance of ModelSelection. Call the parent with null right now. 
		 */
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
		 * This method initializes the attributes. 
		 * Populates the required tables before the tree construction according to the paper. 
		 */
		AttributeHelper.initializeAttributes(data);
		
		/**
		 * Override and invoke the build tree method of the ClassifierTree class
		 */
		buildTree(data, true);
		
		//TODO implement pruning. 
		/**
		 * Pruning methodology: http://ai.stanford.edu/~ronnyk/treesHB.pdf 
		 */
		//TreePruner.pruneTree(this);
	}
	
	/**
	 * Overriding the core buildTree method of the WEKA framerowk. This is the method that will be recursively called
	 * with a qualifying subset of the parent nodes data for the subtree construction. 
	 */
	@Override
	public void buildTree(Instances data, boolean arg1) throws Exception {
		
		/**
		 * m_localModel is the variable which has to be set. This is the one which will have the split. 
		 * Begin with initializing the tree to have no split over the data. Later, choose a split attribute and grow the 
		 * subtrees based on the split subsets. 
		 */
		m_localModel = new NoSplit(new Distribution(data));
		m_train = data;
		
		/**
		 * Calculate the Attribute entropies and IIGs here. 
		 */
		AttributeHelper.calculateAttributeEntropies(data);
		
		int splitAttributeIndex = -1;
		boolean subTreeConstructionPossible = false;
		
		/**
		 * Has to be in an infinite loop because the paper says that it cannot select an 
		 * attribute which has just one split. 
		 */
		while(true){

			splitAttributeIndex = AttributeHelper.getAttributeIndexWithMaxIIGFromData(data);
			/**
			 * When all the attributes are deleted, we end up in a leaf node. This time we don't get any attribute
			 * over which we can split. 
			 */
			if(splitAttributeIndex == -1)
				break;
			
			C45Split currentNodeSplit = new C45Split(splitAttributeIndex, 2, new Distribution(data).total());
			currentNodeSplit.buildClassifier(data);

			/**
			 * remove the current attribute for the children. so that they don't consider it.
			 */
			AttributeHelper.markAttributeDeleteStatus(splitAttributeIndex, true);
			
			if(currentNodeSplit.numSubsets() > 1){
				m_localModel = currentNodeSplit;
				subTreeConstructionPossible = true;
				break;
			}
		}
		
		/**
		 * Build the subtrees now. If the currentNodeSplit.numSubsets == 1, then this is a leaf node
		 * 
		 * Used the subTreeConstructionPossible variable because several instances were seen where no attribute was chosen 
		 * in the previous loop. 
		 */
		if(subTreeConstructionPossible) {
			
			/**
			 * There are going to be currentNodeSplit.numSubsets() subtrees for this node. Construct each of them.
			 * 
			 * Create data subsets for each split
			 */
			Instances [] subtreeSplitData = m_localModel.split(data);
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
		
			for(int i = 0; i < m_localModel.numSubsets(); i ++){
				NaiveDecisionTree newSubTree = new NaiveDecisionTree();
				newSubTree.buildTree(subtreeSplitData[i], true);
				m_sons[i] = newSubTree;
			}
		} else {
			//there has to be just one possible split. Mark this as the leaf.
			m_isLeaf = true;
		}
	}	
}
