package edu.uncc.ml.naivetree.attributes.impl;

import java.util.Enumeration;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AbstractDataAttribute;
import edu.uncc.ml.naivetree.attributes.Constants;
import edu.uncc.ml.naivetree.attributes.ContinuousDataAttribute;

public class ContinuousDataAttributeImpl extends AbstractDataAttribute implements ContinuousDataAttribute {

	private double [] binCutoffs;
	
	public ContinuousDataAttributeImpl(Instances data, int attributeIndex){
		super();
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
		
		if(binCutoffs == null)
			return null;
		
		Instance currentInstance;
		Distribution distribution = new Distribution(binCutoffs.length + 1, instances.numClasses());
		
		double[] bagWeights = new double[distribution.numBags()];
		for (int i = 0; i < distribution.numBags(); i++)
			bagWeights[i] = 1.0 / distribution.numBags();
		
		Enumeration<Instance> instanceList = instances.enumerateInstances();
		
		while (instanceList.hasMoreElements()) {
			currentInstance = (Instance) instanceList.nextElement();
			try {
				if (!currentInstance.isMissing(attributeIndex))
					distribution.add(getBelongingBinIndex(currentInstance, attributeIndex), currentInstance);
				else {
					distribution.addWeights(currentInstance, bagWeights);
				}
			} catch(Exception e){
				e.printStackTrace();
				return null;
			}
		}

		return distribution;
	}
	
	private int getBelongingBinIndex(Instance currentInstance, int attributeIndex) {
		
		if (currentInstance.value(attributeIndex) <= binCutoffs[0])
			return 0;
		
		if (currentInstance.value(attributeIndex) >= binCutoffs[binCutoffs.length - 1])
			return binCutoffs.length;
		
		int index = -1;
		
		for (int i = 0; i < binCutoffs.length - 1; i++) {
			if (currentInstance.value(attributeIndex) >= binCutoffs[i] && currentInstance.value(attributeIndex) <= binCutoffs[i + 1]) {
				index = i + 1;
				break;
			}
		}
		
		return index;
	}

	/**
	 * The algorithm used to bin continuous data attribute values is: Equal Widths: 
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
		double distributionMaximum = 0;
		double distributionMinimum = 1;

		Instance currentInstance;
		for (int i = 0; i < data.numInstances(); i++) {
			currentInstance = data.instance(i);
			if (!currentInstance.isMissing(attributeIndex)) {
				double currentFeatureValue = currentInstance.value(attributeIndex);
				if (distributionMaximum < distributionMinimum) {
					distributionMaximum = distributionMinimum = currentFeatureValue;
				}
				else if (currentFeatureValue > distributionMaximum) {
					distributionMaximum = currentFeatureValue;
				}
				else if (currentFeatureValue < distributionMinimum) {
					distributionMinimum = currentFeatureValue;
				}
			}
		}

		double binSize = (distributionMaximum - distributionMinimum) / bins;
		
		if ((bins > 1) && (binSize > 0)) {
			binCutoffs = new double[bins - 1];
			for (int i = 1; i < bins; i++) {
				binCutoffs[i - 1] = distributionMinimum + binSize * i;
			}
		} else {
			System.out.println("No Bins");
		}
	}
}
