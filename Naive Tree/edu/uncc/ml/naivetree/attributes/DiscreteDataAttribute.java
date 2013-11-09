package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;

/**
 * This interface adds functionality to <code>Discrete Data</code>. 
 * @author Ankur
 */

public interface DiscreteDataAttribute extends DataAttribute {
	
	/**
	 * This method creates a distribution of all the instances in <code>instances</code> into the 
	 * possible discrete bins.
	 * @param instances Input data
	 * @param attributeIndex index of the discrete valued attribute in the data. 
	 * @return <code>Distribution</code> of data in discrete valued bags. 
	 */
	Distribution calculateDistributionForDiscreteBags(Instances instances, int attributeIndex);
	
}
