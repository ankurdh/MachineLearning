package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;

/**
 * The DataAttribute interface defines a general layout of the functionalities that any attribute must
 * have with respect to the decision tree building.
 * 
 * @author Ankur
 *
 */

public interface DataAttribute {
	/**
	 * This method initializes the attribute at index <code>attributeIndex</code> in the data defined by
	 * <code>Instances</code> Should do the following:
	 * 
	 * - analyze the bags and distribute instances into them
	 * - analyze classes and distribute instances into them
	 * 
	 * @param data the dataset
	 * @param attributeIndex the index of the attribute in the dataset to be analyzed
	 */
	void initializeAttribute(Instances data, int attributeIndex);
	
	/**
	 * This method must calculate the <code>Independent Information Gain</code> for the given attribute 
	 * using the formulae [(2) & (5)] defined in the paper
	 * 
	 * The Entropy IIG formula to be implemented is:
	 * ----------------------------------------------------------------------------------------------------------------------------------------------
	 * 								IIG(S, X) = Entropy(S) - SUM_OVER_ALL_ATTRIBUTE_VALUES [|Sx|/|S| * Entropy(Sx)]
	 * ----------------------------------------------------------------------------------------------------------------------------------------------
	 * @param dataEntropy Overall entropy of the dataset
	 * @param classPriors priors of each of the classes
	 */
	void calculateCurrentAttributeIIG(double dataEntropy, double [] classPriors);
	
	/**
	 * Below are definitions of general properties of attributes.
	 */
	
	void setDistribution(Distribution distribution);
	void setAttributeIndex(int index);	
	int getAttributeIndex();
	double getAttributeIIG();
	void setAttributeIIG(double attributeIIG);
	
	void setDeleted(boolean isDeleted);
	boolean isAttributeDeleted();
	
}
