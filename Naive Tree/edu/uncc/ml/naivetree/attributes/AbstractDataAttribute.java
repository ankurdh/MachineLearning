package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Utils;

public abstract class AbstractDataAttribute implements DataAttribute {
	private Distribution dataDistribution;
	private int attributeIndex;
	private double attributeIIG;
	
	private boolean isDeleted;
	
	public AbstractDataAttribute(){
		attributeIndex = -1;
		setAttributeIIG(-1.0);
		
		isDeleted = false;
	}

	@Override
	public void calculateCurrentAttributeIIG(double dataEntropy, double[] classPriors) {

		if(dataDistribution == null || isDeleted)
			setAttributeIIG(0.0);
		else
			setAttributeIIG(dataEntropy - getClassAttributeEntropiesPerValue(classPriors));

	}
	
	/**
	 * This method is basically the implementation of second part of the formula:
	 * <code>IIG(S, X) = Entropy(S) - SUM_OVER_ALL_ATTRIBUTE_VALUES [|Sx|/|S| * Entropy(Sx)]</code><br/>
	 * 
	 * The conditionally independent attributes reduces the value: Entropy(Sx) to P(Ci|Xp) * P(X | C)/SUM_OVER_ALL_CLASSES [(P(Ci|Xp) * P(X | C))]
	 * 
	 * And the value |Sx|/|S| is approximated to the denomiator value: SUM_OVER_ALL_CLASSES [(P(Ci|Xp) * P(X | C))]
	 * 
	 * @param dataDistribution The data. This dataset recursively reduces to dataset that satisfies the conditions in parents of the current node
	 * @param classPriors P(Ci | Xp) are class priors. These class priors are recursively calculated and passed as parameters to this method
	 * @return the second part of the Entropy formula
	 */

	private double getClassAttributeEntropiesPerValue(double[] classPriors) {

		/**
		 * The required parameters for the formula and how they're derived:
		 * |Sx|/|S| is approximated to the denomiator value: SUM_OVER_ALL_CLASSES [(P(Ci|Xp) * P(X | C))]
		 * P(Ci | Xp) are class priors. These class priors are recursively calculated and passed as parameters to this method
		 * P(Xp | Ci) denotes the percentage of data that belongs to class Ci having the attribute X = Xp. This is returned by the method
		 * perClassPerBag() method in the Weka framework.
		 */
		
		/**
		 * The entropy of the overall subset of the data that's passed by recursive subdivision
		 */
		double entropySx = 0;
		
		/**
		 * This variable models: |Sx|/|S|
		 */
		double subsetToDataSizeRatio = 0;
		
		/**
		 * This loop iterates for each of the attribute values for the given attributes. 
		 */
		for (int bagIndex = 0; bagIndex < dataDistribution.numBags(); bagIndex++) {
			/**
			 * Fix the bug where a distribution bag has no elements in the data falling into it. 
			 * Can occur because data is recursively subdivided and filtered.
			 */
			
			if (dataDistribution.perBag(bagIndex) == 0)
				continue;
			
			boolean atleastOneClassProbabilityExists = false;
			
			/**
			 * the classProbabilities variable models Ps(Ci)
			 */
			double[] classProbabilities = new double[dataDistribution.numClasses()];
			double entropyDenominator = 0;
			for (int classIndex = 0; classIndex < dataDistribution.numClasses(); classIndex++) {
				/**
				 * Fixing the bug where the Utils.normalize call failed because of 0 classPriors. This results in a 0 class probability. 
				 */
				if (classPriors[classIndex] == 0 || dataDistribution.perClassPerBag(bagIndex, classIndex) == 0)
					continue;
				
				classProbabilities[classIndex] = dataDistribution.perClassPerBag(bagIndex, classIndex) * classPriors[classIndex] / (dataDistribution.perClass(classIndex));
				
				if(classProbabilities[classIndex] > 0.0)
					atleastOneClassProbabilityExists = true;
				
				entropyDenominator += classProbabilities[classIndex];
			}
			
			try {
				
				if(atleastOneClassProbabilityExists)
					Utils.normalize(classProbabilities);
				else
					return 0;
			} catch (Exception e) {
				e.printStackTrace();
			}

			double runningEntropy = 0;
			for (int i = 0; i < dataDistribution.numClasses(); i++) {
				if (classProbabilities[i] == 0)
					continue;
				runningEntropy -= classProbabilities[i] * (Math.log(classProbabilities[i])/Constants.LOG_2);
			}
			entropySx += entropyDenominator * runningEntropy;
			subsetToDataSizeRatio += entropyDenominator;
		}
		
		if (subsetToDataSizeRatio == 0 || entropySx == 0) {
			return 0;
		}
		
//		System.out.println("IIG: " + entropySx/subsetToDataSizeRatio);
		return entropySx / subsetToDataSizeRatio;		

	}

	@Override
	public void setDistribution(Distribution distribution) {
		if(distribution != null)
			dataDistribution = distribution;
	}

	@Override
	public void setAttributeIndex(int index) {
		attributeIndex = index;
	}
	
	@Override
	public int getAttributeIndex(){
		return attributeIndex;
	}

	@Override
	public double getAttributeIIG() {
		return attributeIIG;
	}

	@Override
	public void setAttributeIIG(double attributeIIG) {
		this.attributeIIG = attributeIIG;
	}
	
	@Override
	public void setDeleted(boolean b){
		isDeleted = b;
	}
	
	@Override
	public boolean isAttributeDeleted(){
		return isDeleted;
	}
}
