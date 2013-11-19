package edu.uncc.ml.naivetree.attributes;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;
import weka.core.Utils;

public abstract class AbstractDataAttribute implements DataAttribute {
	private Distribution dataDistribution;
	private int attributeIndex;
	private double attributeIIG;
	
	private boolean isDeleted;
	
	public AbstractDataAttribute(){
		attributeIndex = -1;
		setAttributeIIG(-1.0);
		dataDistribution = null;
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
		 * 
		 * The entropy of the overall subset of the data that's passed by recursive subdivision
		 * 
		 */
		
		/**
		 * 
		 * Entropy(Sx) =  -For each class: 	P(SxCi) * log_base_2(P(SxCi))
		 * 
		 * Algorithm outline:
		 * 
		 * entropySx = 0.0 
		 * 
		 * for each bag 'b' do
		 *  for each class 'c' do:
		 *   denominator += perClassPerBag(b, c)*classPrior(c)
		 *   
		 * 	For each class 'c' do
		 *   numerator = classPrior(c) * perClassPerBag(b, c)
		 *   pSxCi = numerator/denominator;  
		 *   entropySx -= [pSxCi * log_base_2(pSxCi)]
		 *   
		 *   entropySx *= denominator
		 * 
		 */
		double newEntropySx = 0.0;
		
		for(int bagIndex = 0 ; bagIndex < dataDistribution.numBags(); bagIndex++){
			double numerator = 0.0;
			double denominator = 0.0;
			
			if (dataDistribution.perBag(bagIndex) == 0)
				continue;
			
			for(int classIndex = 0 ; classIndex < dataDistribution.numClasses(); classIndex ++){
				
				if(dataDistribution.perClass(classIndex) == 0 || dataDistribution.perClassPerBag(bagIndex, classIndex) == 0 || classPriors[classIndex] == 0)
					continue;
				
				denominator += (dataDistribution.perClassPerBag(bagIndex, classIndex) * classPriors[classIndex] / dataDistribution.perClass(classIndex));
			}
			
			if(denominator == 0)
				continue;
			
			double runningEntropy = 0.0;
			
			for(int classIndex = 0 ; classIndex < dataDistribution.numClasses(); classIndex ++ ){
				
				if(dataDistribution.perClass(classIndex) == 0 || dataDistribution.perClassPerBag(bagIndex, classIndex) == 0 || classPriors[classIndex] == 0)
					continue;
				
				numerator = (classPriors[classIndex] * dataDistribution.perClassPerBag(bagIndex, classIndex) / dataDistribution.perClass(classIndex));
				
				if(numerator == 0)
					continue;
				
				double pSxCi = numerator/denominator;
				runningEntropy -= (pSxCi * Utils.log2(pSxCi));
			}
			
			newEntropySx += (runningEntropy * denominator);
		}
		
//		System.out.println("New Entropy: " + newEntropySx);
		return newEntropySx;

	}
	
	@Override
	public double [] getMissingAttributeDoubles(Instances instances, int distributionbags){
		
		double[] missingAttributeDoubles = new double[instances.numAttributes()];
		for (int i = 0; i < distributionbags; i++)
			missingAttributeDoubles[i] = 1.0 / distributionbags;
		
		return missingAttributeDoubles;
		
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
