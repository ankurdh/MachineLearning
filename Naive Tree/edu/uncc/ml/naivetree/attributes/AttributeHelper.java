package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.impl.ContinuousDataAttributeImpl;
import edu.uncc.ml.naivetree.attributes.impl.DiscreteDataAttributeImpl;

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

	/**
	 * This method first calculates the value: P(Ci | Xp) for each of the classes. Since the data is recursively subdivided based on the conditions
	 * of the parent nodes, new class probabilities are calculated everytime the method is executed. 
	 * @param data
	 */
	public static void calculateAttributeEntropies(Instances data, DataAttribute [] attributes) {
		try {
			Distribution distribution = new Distribution(data);
			double entropyOfWholeData = 0.0;
			double classPriors[] =  new double[data.numClasses()];
			/**
			 * First iterate through all the classes and find out P(Ci | Xp)
			 * Also calculate the Entropy(S) = -SUM_OVER_ALL_CLASSES(P(Ci | Xp) * log(P(Ci | Xp)))
			 */
			for(int classIndex = 0 ; classIndex < data.numClasses(); classIndex ++){
				/**
				 * For each class, the class prior defined by P(Ci | Xp) is the number of instances belonging to the class Ci. 
				 * We don't have to explicitly check for all the attributes obeying the conditions in Xp because the 
				 * data passed is already run through the tree through all the parent nodes and the Xp attributes have been satisfied.
				 */
				classPriors[classIndex] = distribution.perClass(classIndex)/data.numInstances();
				
				entropyOfWholeData -= classPriors[classIndex] * (Math.log(classPriors[classIndex])/Constants.LOG_2);
				
			}
			System.out.println("Entropy of whole data: " + entropyOfWholeData);
			
			for(int attributeIndex = 0 ; attributeIndex < data.numAttributes(); attributeIndex ++){
				//omit the class index
				if(attributeIndex == data.classIndex())
					continue;
				
				attributes[attributeIndex].calculateCurrentAttributeIIG(entropyOfWholeData, classPriors);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
