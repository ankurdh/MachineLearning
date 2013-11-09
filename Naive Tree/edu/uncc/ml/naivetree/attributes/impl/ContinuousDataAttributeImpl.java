package edu.uncc.ml.naivetree.attributes.impl;

import java.util.Enumeration;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import edu.uncc.ml.naivetree.attributes.AbstractDataAttribute;
import edu.uncc.ml.naivetree.attributes.ContinuousDataAttribute;

public class ContinuousDataAttributeImpl extends AbstractDataAttribute implements ContinuousDataAttribute {

	private double [] binCutoffs;
	
	public ContinuousDataAttributeImpl(Instances data, int attributeIndex){
		
		super.setIndex(attributeIndex);
		
		initializeAttribute(data, attributeIndex);
	}
	
	@Override
	public void initializeAttribute(Instances data, int attributeIndex) {
		int bin = (int) Utils.log2(data.numInstances());
		
		super.setDistribution(calculateCutPointsForBins(data, attributeIndex, bin));

	}

	@SuppressWarnings("unchecked")
	@Override
	public Distribution calculateCutPointsForBins(Instances instances, int attributeIndex, int bins) {
		
		setupCutPointsByEqualWidths(instances, attributeIndex, bins);
		
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
					distribution.add(getBinIndex(currentInstance, attributeIndex), currentInstance);
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
	
	private int getBinIndex(Instance currentInstance, int attributeIndex) {
		
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
		
		if (index == -1)
			index = -1;
		
		return index;
	}

	private void setupCutPointsByEqualWidths(Instances data, int attributeIndex, int bins){
		double maximum = 0;
		double minimum = 1;

		Instance currentInstance;
		for (int i = 0; i < data.numInstances(); i++) {
			currentInstance = data.instance(i);
			if (!currentInstance.isMissing(attributeIndex)) {
				double currentVal = currentInstance.value(attributeIndex);
				if (maximum < minimum) {
					maximum = minimum = currentVal;
				}
				else if (currentVal > maximum) {
					maximum = currentVal;
				}
				else if (currentVal < minimum) {
					minimum = currentVal;
				}
			}
		}

		double binWidth = (maximum - minimum) / bins;
		
		if ((bins > 1) && (binWidth > 0)) {
			binCutoffs = new double[bins - 1];
			for (int i = 1; i < bins; i++) {
				binCutoffs[i - 1] = minimum + binWidth * i;
			}
		}
	}
}
