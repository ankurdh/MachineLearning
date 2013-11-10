package edu.uncc.ml.naivetree.attributes;

import edu.uncc.ml.naivetree.attributes.impl.ContinuousDataAttributeImpl;
import edu.uncc.ml.naivetree.attributes.impl.DiscreteDataAttributeImpl;
import weka.core.Instances;

public class AttributeHelper {

	public static DataAttribute[] initializeAttributes(Instances data, DataAttribute[] attributes) {
		//initialize the attributes array first. Since one value in the attributes is the class, 
		//create an array of size: noOfAttributes - 1
		if(attributes == null)
			attributes = new DataAttribute[data.numAttributes() - 1];
		
		for (int i = 0; i < data.numAttributes(); i++) {
			if (i == data.classIndex())
				continue;
			
			if(data.attribute(i).isNominal())
				attributes[i] = new DiscreteDataAttributeImpl(data, i);
			else if(data.attribute(i).isNumeric())
				attributes[i] = new ContinuousDataAttributeImpl(data, i);
		}
		
		return attributes;
	}
}
