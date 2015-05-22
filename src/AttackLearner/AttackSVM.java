package AttackLearner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;
import Utility.Utility;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import de.bwaldvogel.liblinear.Train;


public class AttackSVM extends AttackLinearLearner{
	private Problem problem;
	private Parameter parameter;
	private Model model;
	private int regNorm;
	private Matrix gradImpFunc;

	@Override
	double calcLoss(int dataId, HashMap<Integer, Double> weights) {
		// TODO Auto-generated method stub
		double predLabel = prediction( curX.get(dataId), weights );
		// reset the true label 0,1 to -1,+1
		int trueLabel = Utility.getLabel( problem.y[dataId] );		
		return Math.max( 0 , 1 - trueLabel*predLabel );
	}

	@Override
	public void initData(String DATA_PATH, String targetWeightFileName, int regNorm2) throws NumberFormatException, IOException, InvalidInputDataException {

		// read the target weights
		BufferedReader finTargetWeight = new BufferedReader( new FileReader( targetWeightFileName ));
		targetWeights = new HashMap<Integer, Double>();
		String line = "";
		while( (line = finTargetWeight.readLine()) != null ) {
			String str[] = line.split(":");
			targetWeights.put( Integer.parseInt(str[0]), Double.parseDouble(str[1]) );
		}
		finTargetWeight.close();
				
		// read the training data
		List<File> trainFiles = new ArrayList<File>();
		trainFiles.add(  new File(DATA_PATH) );
		problem = Train.readProblemFromFiles( trainFiles , 0 );
		System.out.println(problem.n);
		System.out.println(problem.l);
		originX = new ArrayList<HashMap<Integer,Double> >();
		curX = new ArrayList<HashMap<Integer, Double> >();
		for(int i = 0; i < problem.l; ++i) {
			originX.add( new HashMap<Integer,Double>() );
			// feature for bias term
			originX.get(i).put(0, 1.0);
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
		regNorm = regNorm2;

		double C = 1000, eps = 0.01;
		parameter = new Parameter(SolverType.L2R_L1LOSS_SVC_DUAL, C, eps);
		System.out.println( "PARAMETERS: C = " + C);
	}

	@Override
	public void learnModel(String MODEL_PATH, String MODEL_NAME) throws IOException {
		// load the data from curX feature matrix to LibSVM
		for(int dataId = 0; dataId < problem.x.length; dataId++)
			for(int fId = 0; fId < problem.x[dataId].length; fId++)
				problem.x[dataId][fId].setValue( curX.get(dataId).get(problem.x[dataId][fId].getIndex()) );
		
		// call LibSVM to learn the model
		model = Linear.train(problem, parameter);
		
		// load the learned model from LibSVM to learnedWeights
		learnedWeights.clear();
		learnedWeights.put(0, model.getBias());
		for(int i = 0; i < model.getFeatureWeights().length; ++i)
			learnedWeights.put(i+1, model.getFeatureWeights()[i]);
			
		double loss = 0.0;
		double errorRate = 0.0;
		for(int i = 0; i < problem.l; ++i) {
			double lossInstance = calcLoss(i, learnedWeights);
			loss += lossInstance;
			if( lossInstance > Math.log(2) )
				errorRate += 1;
		}
		System.out.println(loss + " " + errorRate/problem.l);
		model.save(new File(MODEL_PATH + MODEL_NAME + "_" + "all"));
	}

	@Override
	public void takeGradient(double stepLength,
			double regularization, BufferedWriter fout) {
		for(int dataId = 0; dataId < curX.size(); dataId++)
			for(int fId : curX.get(dataId).keySet()) {
				// TODO Auto-generated method stub
				int label = Utility.getLabel( problem.y[dataId] );
				double pred = this.prediction(curX.get(dataId), learnedWeights);
						
				// for L2 norm regularized SVM
				if( 1-pred*label > 0 ) {
					double delta = parameter.getC()*weightDis(fId)*label 
									+ regularization*(curX.get(dataId).get(fId) - originX.get(dataId).get(fId));
					double newValue = curX.get(dataId).get(fId) - stepLength * delta;				
					curX.get(dataId).put(fId, newValue);
				}
			}
		calcEffort();
	}

	@Override
	public double getLossOriginX() {
		double loss = 0.0;
		for(int dataId = 0; dataId < problem.l; ++dataId)
			loss += Math.max(0, 1 - labels.get(dataId)*prediction( originX.get(dataId), learnedWeights ) );
		return loss;
	}
}
