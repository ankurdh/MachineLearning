package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;

/**
 * This interface adds functionality to <code>Numeric Data</code>. 
 * @author Ankur
 */

public interface ContinuousDataAttribute extends DataAttribute {

	/**
	 * This meethod calculates thresholds for bins for numeric attributes.
	 * @param instances The data
	 * @param attributeIndex the index of the numeric attribute that is under consideration 
	 * @return a <code>Distrubution</code> of thresholds for bins of numeric attribute bags
	 */
	Distribution calculateCutPointsForBins(Instances instances, int attributeIndex, int bin);
	
}
