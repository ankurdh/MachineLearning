package edu.uncc.ml.naivetree;

import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class Main {

	/**
	 * @param args The standard WEKA arguments.
	 * 
	 * The relevent ones for us is the "-t" option which specifies the input dataset. 
	 */
	public static void main(String[] args) {
		
		Scanner in = null; 
		
		try {
			
			in = new Scanner(System.in);
			System.out.print("Run with new classifier?(Y/N)\nY--> Run with Naive Tree\nN-->Runs with C4.5\nYour Choice: ");
			String c = in.next();
			String classifierOutput = null;
			
			if(c.equalsIgnoreCase("y"))
				classifierOutput = Evaluation.evaluateModel(new NaiveClassifier(), args);
			else 
				classifierOutput = Evaluation.evaluateModel(new J48(), args);
			
			System.out.println(classifierOutput);
			
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if(in != null)
				in.close();
		}
	}
}