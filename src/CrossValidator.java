import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import Utility.Printer;
import AttackLearner.AttackLinearLearner;
import AttackLearner.AttackLogisticRegression;
import AttackLearner.AttackMedianRegression;
import AttackLearner.AttackOrdinaryLeastSquare;
import AttackLearner.AttackRobustSVM;
import AttackLearner.AttackSVM;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Predict;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.Train;

public class CrossValidator {
	
	static String DATA_PATH = "";
	public final String MODEL_PATH = "Input/Model/";
	public final String ANSWER_PATH = "Input/Answer/";
	public final String MODEL_NAME = "model";
	
	public CrossValidator() {
	}

	/*
	 * usage:
	 * args[0] input file path
	 */
	public static void main(String[] args) throws IOException {
		CrossValidator cv = new CrossValidator();
		DATA_PATH = args[0];
		BufferedWriter fout = new BufferedWriter( new FileWriter( args[1] ) );
		String targetWeightFile = args[2];
		String attackMethod = args[3];
		double budget = Double.parseDouble(args[4]);
		int regNorm = Integer.parseInt(args[5]);
		int numIter = Integer.parseInt(args[6]);
		String typeLearner = args[7];
		
		try {	
			cv.gradientDescent(budget, fout, targetWeightFile, regNorm, attackMethod, numIter, typeLearner);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		fout.close();
	}
	
	public void gradientDescent( double budget, BufferedWriter fout, String targetWeightFileName, 
								int regNorm, String attackMethod, int numIter, String typeLearner) throws Exception {
		// define the printer of running result
		Printer printer = new Printer();
		
		// the desired weight vector
		AttackLinearLearner attacker;
		switch (typeLearner) {
			case "AttackSVM":
				attacker = new AttackSVM();
				System.out.println("AttackSVM");
				break;
			case "AttackMedianRegression":
				attacker = new AttackMedianRegression();
				System.out.println("AttackMedianRegression");
				break;
			case "AttackOrdinaryLeastSquare":
				attacker = new AttackOrdinaryLeastSquare();
				System.out.println("AttackOrdinaryLeastSquare");
				break;
			case "AttackLogisticRegression":
				attacker = new AttackLogisticRegression();
				System.out.println("AttackLogisticRegression");
				break;
			case "AttackRobustSVM":
				attacker = new AttackRobustSVM();
				((AttackRobustSVM) attacker).setAttackBudget( budget );
				System.out.println("AttackRobustSVM");
				break;
			default:
				attacker = new AttackSVM();
				break;
		}
		
		attacker.initData(DATA_PATH, targetWeightFileName, 2);
		attacker.normalizeDataAndTargetWeights();
		printer.printWeights(fout, attacker);
		
		/*
		if( attackMethod.equals( "naive" ) || attackMethod.equals( "optimal" ) ) {
			double norm = 0.0;
			for(int j = 0; j < linearModel.targetWeightIdx.size(); j++)
				norm += Math.pow(linearModel.targetWeights.get(j), 2.0);
			//norm = Math.pow(norm, 0.5);
			for(int i = 0; i < linearModel.numDatePoints(); i++) {
				double loss = Math.max(0, linearModel.calcTargetWeightLoss(i)-0.1);
			//	fout.write( "TargetWeightloss: " + loss );
				for(int j = 0; j < linearModel.targetWeightIdx.size(); j++){
					linearModel.takeGradient(i, linearModel.targetWeightIdx.get(j)+1, -loss*linearModel.targetWeights.get(j)/norm, 0.0, fout);
				}
			//	fout.write( "Current: " + linearModel.calcTargetWeightLoss(i) + "\n" );	
			}
			
			linearModel.learnWeights(MODEL_PATH, MODEL_NAME);
 			//fout.write( " " + linearModel.getL1Effort() + " " + linearModel.getL2Effort() + " " + linearModel.weightL2Dis() + "\n");		
			for(int i = 0; i < linearModel.targetWeights.size(); i++)
			{			
		//		fout.write( "Weight " + linearModel.getWeight(i) + "target " + linearModel.targetWeights.get(i) + "\n" );
			}
		}
		*/
		if( attackMethod.equals( "optimal" ) ) {	
			fout.write( "Iter: " + 0 + " " + attacker.getL1Effort() + " " + attacker.getL2Effort() + " " + attacker.weightDis(2) + " " + attacker.getLossOriginX() + "\n");
			
			for(int iter = 0; iter < numIter; iter++) {
				System.out.println("Iter" + iter);

				// calculate gradient
				double stepLength = 60.0 / Math.min(iter+1,40);
				attacker.learnModel( MODEL_PATH, MODEL_NAME);
				attacker.takeGradient(stepLength, budget, fout);
				printer.printLossEffort(fout, iter, attacker);
				printer.printWeights(fout, attacker);
			}
			
			attacker.recoverDatasetFromMeanAndNorm();
			printer.initWriter( "optimalAttackSVMNegativeLabels.txt" );
			printer.printFeatureModification(-1, attacker);
			printer.closeWriter();
			
			printer.initWriter( "optimalAttackSVMPositiveLabels.txt" );
			printer.printFeatureModification(1, attacker);
			printer.closeWriter();
			
			//printer.initWriter( "optimalAttackMedianRegressionResponse.txt" );
			printer.initWriter( "optimalAttackOLS.txt" );
			printer.printResponseModification(attacker, 1);
			printer.closeWriter();
			
			printer.initWriter( "dataBeforeAttack.txt" );
			printer.printData(attacker, true);
			printer.closeWriter();
		
			printer.initWriter( "DataAfterAttack.txt" );
			printer.printData(attacker, false);
			printer.closeWriter();
		}
	}
}
