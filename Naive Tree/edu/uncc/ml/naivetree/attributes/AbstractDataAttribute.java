package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Utils;

public abstract class AbstractDataAttribute implements DataAttribute {
	private Distribution dataDistribution;
	private int attributeIndex;

	@Override
	public double getCurrentAttributeIIG(double dataEntropy, double[] classPriors) {
		if (dataDistribution == null || dataDistribution.actualNumBags() < 2)
			return 0;

		double independentInformationGain = (dataEntropy - (getClassAttributeEntropiesPerValue(dataDistribution, classPriors)));

		return independentInformationGain;
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

	private double getClassAttributeEntropiesPerValue(Distribution dataDistribution, double[] classPriors) {

		/**
		 * The required parameters for the formula and how they're derived:
		 * |Sx|/|S| is approximated to the denomiator value: SUM_OVER_ALL_CLASSES [(P(Ci|Xp) * P(X | C))]
		 * P(Ci | Xp) are class priors. These class priors are recursively calculated and passed as parameters to this method
		 * P(Xp | Ci) denotes the percentage of data that belongs to class Ci having the attribute X = Xp. This is returned by the method
		 * perClassPerBag() method in the Weka framework.
		 */
		
		double entropySx = 0.0;

		for (int classIndex = 0; classIndex < dataDistribution.numClasses(); classIndex++) {

			double denominator = 0.0;

			for (int bagIndex = 0; bagIndex < dataDistribution.numBags(); bagIndex++) {

				double perBagPerClass = dataDistribution.perClassPerBag(bagIndex, classIndex);
				double numerator = perBagPerClass * classPriors[classIndex];

				for (int x = 0; x < dataDistribution.numClasses(); x++)
					denominator += classPriors[x]
							* dataDistribution.perClassPerBag(bagIndex, x);

				/**
				 * pCiXpX is a representative of P(Ci | Xp, X)
				 */
				
				double pCiXpX = numerator / denominator;

				/**
				 * This is just a summation of entropy. Must still multiply with |Sx|/|S| which is approximated to denomonator.
				 */
				entropySx -= pCiXpX;

			}

			entropySx *= denominator;

		}

		return entropySx;

	}

	protected void setDistribution(Distribution distribution) {
		dataDistribution = distribution;
	}

	protected void setIndex(int index) {
		attributeIndex = index;
	}
}
