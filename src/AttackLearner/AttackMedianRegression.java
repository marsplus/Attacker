package AttackLearner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;
import Utility.Printer;
import Utility.Utility;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.Train;

public class AttackMedianRegression extends AttackLinearLearner {
	private Problem problem;
	// regularization parameter
	private double C;
	// the total amount of modification can be done on feature
	private double attackBudget;
	final double ratio = 0.5;
	
	public void setAttackBudget( double a ) {
		attackBudget = a;
	}
	
	@Override
	double calcLoss(int dataId, HashMap<Integer, Double> weights) {
		double pred = prediction( curX.get(dataId), weights );
		// reset the true label 0,1 to -1,+1
		double y = curY.get(dataId);		
		return Math.abs( y - pred );
	}

	public void initData(String DATA_PATH, String targetWeightFileName,
			int regNorm2) throws FileNotFoundException, NumberFormatException,
			IOException, InvalidInputDataException {
		// read the target weights
		BufferedReader finTargetWeight = new BufferedReader( new FileReader( targetWeightFileName ));
		targetWeights = new HashMap<Integer, Double>();
		String line = "";
		while( (line = finTargetWeight.readLine()) != null ) {
			String str[] = line.split(":");
			targetWeights.put( Integer.parseInt(str[0]), Double.parseDouble(str[1]) );
		}
		learnedWeights = (HashMap<Integer, Double>) targetWeights.clone();
		learnedWeights.put(0, 0.0);
		finTargetWeight.close();
				
		// read the training data
		List<File> trainFiles = new ArrayList<File>();
		trainFiles.add(  new File(DATA_PATH) );
		problem = Train.readProblemFromFiles( trainFiles , 0 );
		System.out.println(problem.n);
		System.out.println(problem.l);
		originX = new ArrayList<HashMap<Integer,Double> >();
		curX = new ArrayList<HashMap<Integer, Double> >();
		curY = new ArrayList<Double>();
		originY = new ArrayList<Double>();
		for(int i = 0; i < problem.l; ++i) {
			originX.add( new HashMap<Integer,Double>() );
			// feature for bias term
			originX.get(i).put(0, 1.0);
			curY.add( problem.y[i] );
			originY.add( problem.y[i] );
			for(int j = 0; j < problem.x[i].length-1; j++) {
				originX.get(i).put(problem.x[i][j].getIndex(), problem.x[i][j].getValue());
			}
			curX.add( (HashMap<Integer, Double>) originX.get(i).clone() );
		}
			
		labels = new ArrayList<Integer>();
		for(int i = 0; i < problem.l; ++i)
			labels.add( Utility.getLabel(problem.y[i]) );
		
		L2Effort = 0.0;
		L1Effort = 0.0;

		// set the regularization parameter and the attack attackBudget
		C = 100;
		System.out.println( "PARAMETERS: C = " + C);
	}

	@Override
	public void learnModel(String MODEL_PATH, String MODEL_NAME)
			throws IOException {
		int iter = 0;
		double diff = 1;
		double stepLength = 0.1;
					
		HashMap<Integer, Double> deltaW = new HashMap<Integer, Double>();
		while( iter<1000 && diff>1e-3 ) {
			iter += 1;
			stepLength = Math.max(0.5/iter, 0.05/1000);
			
			// calc the gradient of the robust SVM objective
			for(int fId : learnedWeights.keySet())
				deltaW.put(fId, learnedWeights.get(fId));
			for(int dataId = 0; dataId < problem.l; ++dataId)
				addGradLossInstance(dataId, deltaW, learnedWeights);
			
			// update by the gradient
			diff = 0.0;
			for(int fId : deltaW.keySet()) {
				diff += stepLength*Math.pow(deltaW.get(fId), 2.0);
				learnedWeights.put(fId, learnedWeights.get(fId) - stepLength*deltaW.get(fId));
				if( iter%100==0 ) {
					double totalLoss = 0.0;
					for(int dataId = 0; dataId < problem.l; ++dataId)
						totalLoss += calcLoss(dataId, learnedWeights);	
					System.out.println("Iter" + iter + "Weight" + fId + " " + learnedWeights.get(fId) + "loss" + totalLoss);
				}
			}
		}
	}

	private void addGradLossInstance(int dataId,
			HashMap<Integer, Double> deltaW,
			HashMap<Integer, Double> learnedWeights) {
		double y = curY.get(dataId);
		double pred = prediction(curX.get(dataId), learnedWeights);
		double ratioCurInstance = getRatio(y, pred);

		for(int fId : curX.get(dataId).keySet()) {
			if( deltaW.containsKey(fId) )
				deltaW.put(fId, deltaW.get(fId) - C*ratioCurInstance*curX.get(dataId).get(fId));
			else
				deltaW.put(fId, 				- C*ratioCurInstance*curX.get(dataId).get(fId));					
		}
	}
	

	@Override
	public void takeGradient(double stepLength, double attackBudget, BufferedWriter fout) {
		double regularization = 0.0005; // use a large number as the regularization
		Matrix gradWY = gradImplicitFunction();
		for(int dataId = 0; dataId < curX.size(); dataId++) {
			System.out.println("gradWY = " + gradWY.get(0, dataId));
		}
		for(int dataId = 0; dataId < curX.size(); dataId++) {
			double y = curY.get(dataId);
			double pred = this.prediction(curX.get(dataId), learnedWeights);
			
			// TODO Auto-generated method stub
				
			// for L2 norm regularized SVM
			double delta = 0.0; // C*weightDis(fId)*ratioCurInstance;
			for(int wId : curX.get(dataId).keySet()) 
				if( targetWeights.containsKey(wId) ) {
					delta += weightDis(wId)*gradWY.get(wId-1, dataId);
				}
					
			//if( L1Effort > attackBudget ) { // if the modification exceeds the attackBudget
				//delta += regularization*(curY.get(dataId) - originY.get(dataId));
			//}				
			double newY = Utility.softThreshold( curY.get(dataId) - stepLength * delta, originY.get(dataId), stepLength*regularization);
			curY.set(dataId, newY);
			
			System.out.println( ""+ dataId + "deltaY" + delta + " " + curY.get(dataId) + " " + originY.get(dataId) );
		}
		calcYEffort();
		System.out.println( "L1Effort"+ this.L1Effort + " L2Effort" + this.L2Effort );		
	}
	
	private double getRatio(double y, double pred) {
		// TODO Auto-generated method stub
		if( y - pred > 0 ) {
			return ratio;
		} else {
			return ratio-1;
		}
	}

	@Override
	public double getLossOriginX() {
		double loss = 0.0;
		for(int dataId = 0; dataId < problem.l; ++dataId)
			loss += Math.abs( curY.get(dataId) - prediction( originX.get(dataId), learnedWeights ) );
		return loss;
	}
	
	public Matrix gradImplicitFunction()
	{
		int N = problem.l;
		int D = learnedWeights.size()-1;
	
		Matrix jacob = new Matrix(D, D);
		Matrix gradWY = new Matrix(D, N);
		for(int fId : targetWeights.keySet()) 
			if( targetWeights.containsKey(fId) )
				jacob.set(fId-1, fId-1, 1.0);
	
		for(int dataId = 0; dataId < problem.l; ++dataId) {
			double pred = this.prediction(curX.get(dataId), learnedWeights);
			if( Math.abs(curY.get(dataId)-pred) < 1 ) { // |pred - actual| <= threshold
				for(int fIdRow : targetWeights.keySet()) 
					for(int fIdCol : targetWeights.keySet()) 
						jacob.set(fIdRow-1, fIdCol-1, jacob.get(fIdRow-1, fIdCol-1) + C*curX.get(dataId).get(fIdRow)*curX.get(dataId).get(fIdCol) );

				for(int fId : curX.get(dataId).keySet()) 
					if( targetWeights.containsKey(fId) )
						gradWY.set(fId-1, dataId, -C*curX.get(dataId).get(fId));
			}
		}
		
		Matrix jacobInv = (jacob.inverse()).times(gradWY); // gradient of hat{w} w.r.t. y = jacob^-1 * gradWY
		return jacobInv.times(-1);
	}
}
