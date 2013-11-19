package edu.uncc.ml.naivetree.attributes.impl;

import java.util.Enumeration;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.AbstractDataAttribute;
import edu.uncc.ml.naivetree.attributes.DiscreteDataAttribute;

public class DiscreteDataAttributeImpl extends AbstractDataAttribute implements DiscreteDataAttribute {

	public DiscreteDataAttributeImpl(Instances data, int attributeIndex){
		super.setAttributeIndex(attributeIndex);
		initializeAttribute(data, attributeIndex);
	}
	
	@Override
	public void initializeAttribute(Instances data, int attributeIndex) {
		super.setDistribution(calculateDistributionForDiscreteBags(data, attributeIndex));
	}

	@SuppressWarnings("unchecked")
	@Override
	public Distribution calculateDistributionForDiscreteBags(Instances instances, int attributeIndex) {
		Instance currentInstance;

		Distribution distribution = new Distribution(instances.attribute(attributeIndex).numValues(), instances.numClasses());

		// Only Instances with known values are relevant.
		Enumeration<Instance> enu = instances.enumerateInstances();
		double [] missingAttributeDoubles = getMissingAttributeDoubles(instances, distribution.numBags());
		
		while (enu.hasMoreElements()) {
			currentInstance = enu.nextElement();
			try {
				if(currentInstance.isMissing(attributeIndex))
					currentInstance.replaceMissingValues(missingAttributeDoubles);
				
				distribution.add((int) currentInstance.value(attributeIndex), currentInstance);
			} catch (Exception e){
				e.printStackTrace();
				return null;
			}
		}

		return distribution;
	}
}
