package edu.uncc.ml.naivetree.attributes.impl;

import java.util.Enumeration;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AbstractDataAttribute;
import edu.uncc.ml.naivetree.attributes.Constants;
import edu.uncc.ml.naivetree.attributes.ContinuousDataAttribute;

public class ContinuousDataAttributeImpl extends AbstractDataAttribute implements ContinuousDataAttribute {

	private double [] binThresholds;
	
	public ContinuousDataAttributeImpl(Instances data, int attributeIndex){
		super.setAttributeIndex(attributeIndex);
		initializeAttribute(data, attributeIndex);
	}
	
	@Override
	public void initializeAttribute(Instances data, int attributeIndex) {
		int binsToBeCreated = (int) (Math.log(data.numInstances())/Constants.LOG_2);
		super.setDistribution(getDistsributionFor(data, attributeIndex, binsToBeCreated));
	}

	@SuppressWarnings("unchecked")
	@Override
	public Distribution getDistsributionFor(Instances instances, int attributeIndex, int bins) {
		
		calculateCutPointsByEqualWidthBins(instances, attributeIndex, bins);
		
		if(binThresholds == null)
			return null;
		
		Instance currentInstance;
		Distribution distribution = new Distribution(binThresholds.length + 1, instances.numClasses());
		
		Enumeration<Instance> instanceList = instances.enumerateInstances();
		double [] missingAttributeDoubles = getMissingAttributeDoubles(instances, distribution.numBags());
		
		while (instanceList.hasMoreElements()) {
			currentInstance = (Instance) instanceList.nextElement();
			try {
				if(currentInstance.isMissing(attributeIndex))
					currentInstance.replaceMissingValues(missingAttributeDoubles);
				
				int distributionIndex = -1;
				
				if (currentInstance.value(attributeIndex) <= binThresholds[0]) {
					distributionIndex = 0;
				} else if (currentInstance.value(attributeIndex) >= binThresholds[binThresholds.length - 1]) {
					distributionIndex = binThresholds.length;
				} else {
					for (int i = 0; i < binThresholds.length - 1; i++) 
						if (currentInstance.value(attributeIndex) >= binThresholds[i] && currentInstance.value(attributeIndex) <= binThresholds[i + 1]) {
							distributionIndex = i + 1;
							break;
						}
				}
				
				distribution.add(distributionIndex, currentInstance);
			} catch(Exception e){
				e.printStackTrace();
				return null;
			}
		}

		return distribution;
	}

	/**
	 * The paper defines that the algorithm used to bin continuous data attribute values is: Equal Widths: 
	 * Algorithm Outline:
	 * 
	 * The algorithm divides the data into k intervals of equal size. The width of intervals is:
	 * 		w = (max-min)/k 
	 * 		And the interval boundaries are: min+w, min+2w, ... , min+(k-1)w
	 * 
	 * @param data
	 * @param attributeIndex
	 * @param bins
	 */
	private void calculateCutPointsByEqualWidthBins(Instances data, int attributeIndex, int bins){
		double distributionMaximum = 0.0;
		double distributionMinimum = 1.0;

		/**
		 * Create a distribution for handling missing values.
		 */
		int numBags = 0;
		
		try {
			numBags = new Distribution(data).numBags();
		} catch (Exception e) {
			e.printStackTrace();
			return;
		}
		
		for (int i = 0; i < data.numInstances(); i++) {
			Instance currentInstance = data.instance(i);

			if(currentInstance.isMissing(attributeIndex))
				currentInstance.replaceMissingValues(getMissingAttributeDoubles(data, numBags));
			
			double currentFeatureValue = currentInstance.value(attributeIndex);
			if (currentFeatureValue > distributionMaximum) {
				distributionMaximum = currentFeatureValue;
			} else if (currentFeatureValue < distributionMinimum) {
				distributionMinimum = currentFeatureValue;
			}
		}

		double binSize = (distributionMaximum - distributionMinimum) / bins;
		
		if ((bins > 1) && (binSize > 0)) {
			binThresholds = new double[bins - 1];
			for (int i = 1; i < bins; i++) 
				binThresholds[i - 1] = distributionMinimum + binSize * i;
		} else {
			System.out.println("No Bins");
		}
	}	
}
