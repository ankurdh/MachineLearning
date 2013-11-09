package edu.uncc.ml.naivetree.attributes;

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
	 * @param dataEntropy Overall entropy of the dataset
	 * @param classPriors priors of each of the classes
	 * @return Independent Information Gain of the current attribute
	 */
	double getCurrentAttributeIIG(double dataEntropy, double [] classPriors);
	
}
