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

	private double getClassAttributeEntropiesPerValue(Distribution dataDistribution, double[] classPriors) {

		//TODO modify this code.
		
		double hSonx = 0;
		double sumY = 0;
		
		for (int bagIndex = 0; bagIndex < dataDistribution.numBags(); bagIndex++) {
			if (dataDistribution.perBag(bagIndex) == 0)
				continue;
			
			double[] postProbs = new double[dataDistribution.numClasses()];
			boolean empty = true;
			double tempY = 0;
			for (int i = 0; i < dataDistribution.numClasses(); i++) {
				if (classPriors[i] == 0)
					continue;
				double likelihood = (dataDistribution.perClassPerBag(bagIndex, i) + Constants.LAPLACE_CONSTANT)
						/ (dataDistribution.perClass(i) + Constants.LAPLACE_CONSTANT * dataDistribution.actualNumBags());
				postProbs[i] = likelihood * classPriors[i];
				tempY += postProbs[i];
				if (postProbs[i] > 0)
					empty = false;
			}
			
			if (empty)
				continue;
			try {
				Utils.normalize(postProbs);
			} catch (Exception e) {
				e.printStackTrace();
			}

			double tempEn = 0;
			for (int i = 0; i < dataDistribution.numClasses(); i++) {
				if (postProbs[i] == 0)
					continue;
				tempEn -= postProbs[i] * Utils.log2(postProbs[i]);
			}
			hSonx += tempY * tempEn;
			sumY += tempY;
		}

		if (sumY == 0) {
			return 0;
		}
		hSonx = hSonx / (sumY);
		return (hSonx);

	}

	protected void setDistribution(Distribution distribution){
		dataDistribution = distribution;
	}
	
	protected void setIndex(int index){
		attributeIndex = index;
	}
}
