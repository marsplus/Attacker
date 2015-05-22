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
		C = 10;
		System.out.println( "PARAMETERS: C = " + C);
	}

	@Override
	public void learnModel(String MODEL_PATH, String MODEL_NAME)
			throws IOException {
		int iter = 0;
		double diff = 1;
		double stepLength = 0.1;
		HashMap<Integer, Double> meanX = new HashMap<Integer, Double>();
		HashMap<Integer, Double> normX = new HashMap<Integer, Double>();
		
		// normalize each feature among all datapoints
		for(int fId : targetWeights.keySet()) {
			meanX.put(fId, 0.0);
			normX.put(fId, 0.0);
			
			for(int dataId = 0; dataId < problem.l; ++dataId)
				meanX.put( fId, meanX.get(fId)+curX.get(dataId).get(fId) );
			meanX.put(fId, meanX.get(fId)/problem.l);
			
			for(int dataId = 0; dataId < problem.l; ++dataId) {
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)-meanX.get(fId));
				normX.put(fId, normX.get(fId) + Math.abs(curX.get(dataId).get(fId)) );
			}
			normX.put(fId, normX.get(fId)/problem.l);
			for(int dataId = 0; dataId < problem.l; ++dataId) {
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)/normX.get(fId));
				//System.out.println(curX.get(dataId).get(fId));
			}
		}
			
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
		
		// recover each feature 
		for(int fId : targetWeights.keySet()) {
			for(int dataId = 0; dataId < problem.l; ++dataId)
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)*normX.get(fId) + meanX.get(fId));
			learnedWeights.put(0, learnedWeights.get(0) - learnedWeights.get(fId)*meanX.get(fId)/normX.get(fId) );
			learnedWeights.put(fId, learnedWeights.get(fId)/normX.get(fId) );
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
		double regularization = 0.2; // use a large number as the regularization
		Matrix gradWY = gradImplicitFunction();
		for(int dataId = 0; dataId < curX.size(); dataId++) {
			System.out.println("gradWY = " + gradWY.get(0, dataId));
		}
		for(int dataId = 0; dataId < curX.size(); dataId++) {
			double y = curY.get(dataId);
			double pred = this.prediction(curX.get(dataId), learnedWeights);
			double ratioCurInstance = getRatio(y, pred);
			
			// TODO Auto-generated method stub
				
			// for L2 norm regularized SVM
			double delta = 0.0; // C*weightDis(fId)*ratioCurInstance;
			for(int wId : curX.get(dataId).keySet()) 
				if( targetWeights.containsKey(wId) ) {
					delta += weightDis(wId)*gradWY.get(wId-1, dataId);
				}
					
			if( L1Effort > attackBudget ) { // if the modification exceeds the attackBudget
				delta += regularization*(curY.get(dataId) - originY.get(dataId));
			}				
					
			curY.set(dataId, curY.get(dataId) - stepLength * delta);
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

	private double getNormFeatureDiff(int dataId) {
		double l2normFeatureDiff = 0;
		for(int fId : curX.get(dataId).keySet()) {
			l2normFeatureDiff += Math.pow(curX.get(dataId).get(fId) - originX.get(dataId).get(fId), 2.0);
		}
		return Math.pow(l2normFeatureDiff, 0.5);
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
	
		Matrix jacob = new Matrix(3*N+D, 3*N+D);
		double[][] Xarray = new double[N][D];
		double[] lambda = new double[N];
		Matrix diagLambda1 = new Matrix(N,N);
		Matrix diagLambda2 = new Matrix(N,N);
		Matrix diagUpos = new Matrix(N,N);
		Matrix diagUneg = new Matrix(N,N);
		
		for(int dataId = 0; dataId < problem.l; ++dataId) {
			double pred = this.prediction(curX.get(dataId), learnedWeights);
			if( curY.get(dataId) > pred ) { // u^{+}_i > 0 u^{-}_i = 0
				lambda[dataId] = 0.5*C;
				diagUpos.set(dataId, dataId, curY.get(dataId)-pred);
				diagUneg.set(dataId, dataId, 0 );
			} else { // u^{+}_i = 0 u^{-}_i > 0
				lambda[dataId] = -0.5*C;
				diagUpos.set(dataId, dataId, 0 );
				diagUneg.set(dataId, dataId, pred-curY.get(dataId) );
			}
			
			diagLambda1.set(dataId, dataId, 0.5*C - lambda[dataId] );
			diagLambda2.set(dataId, dataId, 0.5*C + lambda[dataId] );
			
			for(int fId : curX.get(dataId).keySet()) 
				if( targetWeights.containsKey(fId) )
				Xarray[dataId][fId-1] = curX.get(dataId).get(fId);
		}
		Matrix X = new Matrix(Xarray);

		jacob.setMatrix(0, D-1, 0, D-1, new Matrix(D,D));
		jacob.setMatrix(0, D-1, D+2*N, D+3*N-1, (X.transpose()).times(-C) );
		
		jacob.setMatrix(D, D+N-1, D, D+N-1, diagLambda1);
		jacob.setMatrix(D, D+N-1, D+2*N, D+3*N-1, diagUpos.times(-1));
		
		jacob.setMatrix(D+N, D+2*N-1, D+N, D+2*N-1, diagLambda2);
		jacob.setMatrix(D+N, D+2*N-1, D+2*N, D+3*N-1, diagUneg);
		
		jacob.setMatrix(D+2*N, D+3*N-1, 0, D-1, X.times(-1));
		jacob.setMatrix(D+2*N, D+3*N-1, D, D+N-1, (Matrix.identity(N, N)).times(-1));
		jacob.setMatrix(D+2*N, D+3*N-1, D+N, D+2*N-1, Matrix.identity(N, N));
		
		Matrix JacobSquare = ((jacob.transpose()).times(jacob)).plus( Matrix.identity(D+3*N, D+3*N).times(0.01) );
		Matrix jacobInv = (JacobSquare.inverse()).times(jacob.transpose());
		
		Matrix gradWY = jacobInv.getMatrix(0, D-1, D+2*N, D+3*N-1);	
		return gradWY.times(-1);
	}
}
