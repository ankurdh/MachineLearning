package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;
import edu.uncc.ml.naivetree.attributes.impl.ContinuousDataAttributeImpl;
import edu.uncc.ml.naivetree.attributes.impl.DiscreteDataAttributeImpl;

/**
 * The AttributeHelper class models the attributes of the given dataset. The class has an array of
 * attributes defined as the interface DataAttribute. 
 * 
 * Attributes have to be initialized just once for the whole tree construction and thus have
 * been declared as a private static instance in the class. 
 * 
 * @author Ankur
 *
 */
public class AttributeHelper {

	private static DataAttribute [] attributes;
	
	public DataAttribute [] getAttributes(){
		return attributes;
	}
	
	public static DataAttribute[] initializeAttributes(Instances data) {
		/**
		 * initialize the attributes array first.
		 * Since one value in the attributes is the class create an array of size: noOfAttributes - 1 
		 */
		
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
	 * This method first calculates the value: P(Ci | Xp) for each of the classes. Since the data is recursively 
	 * subdivided based on the conditions of the parent nodes, new class probabilities are calculated everytime 
	 * the method is executed. 
	 * @param data recursively subdivided data
	 */
	public static void calculateAttributeEntropies(Instances data) {
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
				//ensure there is atleast one feature for the current class:
				if(distribution.perClass(classIndex) == 0)
					continue;
				
				classPriors[classIndex] = distribution.perClass(classIndex)/data.numInstances();
				
				entropyOfWholeData -= classPriors[classIndex] * (Math.log(classPriors[classIndex])/Constants.LOG_2);
				
			}
//			System.out.println("Entropy of whole data: " + entropyOfWholeData);
			
			for(int attributeIndex = 0 ; attributeIndex < data.numAttributes(); attributeIndex ++){
//				//omit the class index
				if(attributeIndex == data.classIndex())
					continue;
				
				attributes[attributeIndex].calculateCurrentAttributeIIG(entropyOfWholeData, classPriors);
			}
			
			//TODO the below loop has to be removed once the entropies are getting calculated properly.
//			for(int i = 0 ; i < attributes.length; i ++)
//				System.out.print(attributes[i].getAttributeIIG() + " ");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static int getAttributeIndexWithMaxIIGFromData(Instances data){
		
		/**
		 * Now we have attribute IIGs. Iterate over all attributes and return the one with maximum IIG.
		 */
		double currentMaxIIG = 0;
		int maxAttributeIndex = -1;
		
		for(int attributeIndex = 0 ; attributeIndex < data.numAttributes(); attributeIndex++){
			
			if(attributeIndex == data.classIndex())
				continue;
			
			/**
			 * Attributes might not have any data entries qualifying them. This bug occurred when the attribute impurity is 0.  
			 */
			
			if(attributes[attributeIndex].isAttributeDeleted() || attributes[attributeIndex].getAttributeIIG() == -1.0)
				continue;
			
			if(attributes[attributeIndex].getAttributeIIG() > currentMaxIIG){
				currentMaxIIG = attributes[attributeIndex].getAttributeIIG();
				maxAttributeIndex = attributeIndex;
			}
		}
		
		return maxAttributeIndex;
	}

	public static void setAttributeValue(int attributeIndex, DataAttribute newAttribute) {
		attributes[attributeIndex] = newAttribute;
	}

	public static void markAttributeDeleteStatus(int attributeIndex, boolean deleteStatus) {
		attributes[attributeIndex].setDeleted(deleteStatus);
	}
}
