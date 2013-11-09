package edu.uncc.ml.naivetree.attributes.impl;

import java.util.Enumeration;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AbstractDataAttribute;
import edu.uncc.ml.naivetree.attributes.DiscreteDataAttribute;

public class DiscreteDataAttributeImpl extends AbstractDataAttribute implements DiscreteDataAttribute {

	public DiscreteDataAttributeImpl(Instances data, int attributeIndex){
		super.setIndex(attributeIndex);
		
		initializeAttribute(data, attributeIndex);
		
	}
	
	@Override
	public void initializeAttribute(Instances data, int attributeIndex) {
		super.setDistribution(calculateDistributionForDiscreteBags(data, attributeIndex));
	}

	@SuppressWarnings("unchecked")
	@Override
	public Distribution calculateDistributionForDiscreteBags(Instances instances, int attributeIndex) {
		Instance instance;

		Distribution distribution = new Distribution(instances.attribute(attributeIndex).numValues(), instances.numClasses());

		// Only Instances with known values are relevant.
		Enumeration<Instance> enu = instances.enumerateInstances();
		
		double[] bagWeights = new double[distribution.numBags()];
		for (int i = 0; i < distribution.numBags(); i++)
			bagWeights[i] = 1.0 / distribution.numBags();
		
		while (enu.hasMoreElements()) {
			instance = enu.nextElement();
			try {
				if (!instance.isMissing(attributeIndex))
					distribution.add((int) instance.value(attributeIndex), instance);
				else {
					distribution.addWeights(instance, bagWeights);
				}
			} catch (Exception e){
				e.printStackTrace();
				return null;
			}
		}

		return distribution;
	}

}
