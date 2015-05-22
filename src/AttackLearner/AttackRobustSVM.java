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

public class AttackRobustSVM extends AttackLinearLearner {
	private Problem problem;
	// regularization parameter
	private double C;
	// the total amount of modification can be done on feature
	private double attackBudget;
	private Matrix gradImpFunc;

	public void setAttackBudget( double a ) {
		attackBudget = a;
	}
	
	@Override
	double calcLoss(int dataId, HashMap<Integer, Double> weights) {
		double predLabel = prediction( curX.get(dataId), weights );
		// reset the true label 0,1 to -1,+1
		int trueLabel = Utility.getLabel( problem.y[dataId] );		
		return Math.max( 0 , 1 - trueLabel*predLabel );
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
		
		HashMap<Integer, Double> deltaW = new HashMap<Integer, Double>();
		while( iter<1000 && diff>1e-3 ) {
			iter += 1;
			stepLength = Math.max(0.1/iter, 0.1/200);
			
			// calc the gradient of the robust SVM objective
			for(int fId : learnedWeights.keySet())
				deltaW.put(fId, learnedWeights.get(fId));
			addGradL2norm(deltaW, learnedWeights);
			for(int dataId = 0; dataId < problem.l; ++dataId)
				addGradLossInstance(dataId, deltaW, learnedWeights);
			
			// update by the gradient
			diff = 0.0;
			for(int fId : deltaW.keySet()) {
				diff += stepLength*Math.pow(deltaW.get(fId), 2.0);
				learnedWeights.put(fId, learnedWeights.get(fId) - stepLength*deltaW.get(fId));
				if( iter%100==0 ) 
					System.out.println("Iter" + iter + "Weight" + fId + " " + learnedWeights.get(fId));
			}
		}
	}

	private void addGradLossInstance(int dataId,
			HashMap<Integer, Double> deltaW,
			HashMap<Integer, Double> learnedWeights) {
		int trueLabel = Utility.getLabel( problem.y[dataId]);
		double pred = prediction(curX.get(dataId), learnedWeights);
		if( 1 - trueLabel*pred > 0 ) {
			for(int fId : curX.get(dataId).keySet()) {
				if( deltaW.containsKey(fId) )
					deltaW.put(fId, deltaW.get(fId) - C*trueLabel*curX.get(dataId).get(fId));
				else
					deltaW.put(fId, 				- C*trueLabel*curX.get(dataId).get(fId));					
			}
		}
	}
	
	private double getWeightL2Norm( HashMap<Integer, Double> weights ) {
		double l2norm = 0;
		for(int i : weights.keySet())
			l2norm += Math.pow(weights.get(i), 2.0);
		return Math.pow(l2norm, 0.5);
	}
	
	private void addGradL2norm(HashMap<Integer, Double> deltaW,
			HashMap<Integer, Double> learnedWeights) {
		// calc the l2norm (NOT l2norm square) of the learnedWeights
		double l2norm = getWeightL2Norm(learnedWeights);
		// update deltaW by adding gradient of l2norm w.r.t. w
		for(int i : learnedWeights.keySet())
			deltaW.put(i, deltaW.get(i) + attackBudget * learnedWeights.get(i)/l2norm);
	}

	@Override
	public void takeGradient(double stepLength, double attackBudget, BufferedWriter fout) {
		gradImplicitFunction();
		double predOrigin[] = new double[problem.l];
		for(int dataId = 0; dataId<problem.l; ++dataId) {
			predOrigin[dataId] = prediction(originX.get(dataId), learnedWeights);
		}
		// the gradient
		ArrayList<HashMap<Integer, Double> > deltaX = new ArrayList<HashMap<Integer, Double> >();
		// feature sum on each dimension for nonzero hinge loss instances
		HashMap<Integer, Double> fSum = new HashMap<Integer, Double>();
		for(int dataId = 0; dataId < curX.size(); ++dataId) {
			deltaX.add( new HashMap<Integer, Double>() );
			for(int fId : curX.get(dataId).keySet()) {
				deltaX.get(dataId).put(fId, 0.0);
				fSum.put( fId, 0.0 );
			}
		}
				
		// calculate the feature sum for nonzero hinge loss instances
		for(int dataId = 0; dataId < originX.size(); dataId++) 
			if( 1 - labels.get(dataId)*predOrigin[dataId] > 0 ) {
				for(int fId : originX.get(dataId).keySet()) {
					fSum.put( fId, fSum.get(fId) - labels.get(dataId)*originX.get(dataId).get(fId) );
					System.out.println("Yes" + dataId + " " + predOrigin[dataId] + " " + labels.get(dataId) + " " + originX.get(dataId).get(fId) );
				}
			}
		
		// calculate gradient w.r.t. X on loss
		for(int fId : fSum.keySet()) {
			System.out.println("FSUM" + fSum.get(fId) );
			for(int tmpDataId = 0; tmpDataId < curX.size(); ++tmpDataId)
				for(int tmpfId: curX.get(tmpDataId).keySet() )
					deltaX.get(tmpDataId).put(tmpfId, deltaX.get(tmpDataId).get(tmpfId) 
									+ fSum.get(fId)*gradImpFunc.get( fId , tmpfId*problem.l + tmpDataId) );			
		}
		
		// calculate gradient on L2 norm robust SVM TODO: consider the regularization constraint relationship.
		double regularization = 50.0; // use a large number as the regularization
		if( L1Effort > attackBudget ) { // if the modification exceeds the attackBudget
			for(int tmpDataId = 0; tmpDataId < curX.size(); ++tmpDataId) {
				for(int tmpfId: curX.get(tmpDataId).keySet() ) {
					deltaX.get(tmpDataId).put(tmpfId, deltaX.get(tmpDataId).get(tmpfId) 
									- regularization * (curX.get(tmpDataId).get(tmpfId) - originX.get(tmpDataId).get(tmpfId)) );
					System.out.println( "fSum" + fSum.get(tmpfId) + "DeltaX" + tmpDataId + " " + tmpfId + " " + deltaX.get(tmpDataId).get(tmpfId) );
				}
			}
		}
		
		for(int tmpDataId = 0; tmpDataId < curX.size(); ++tmpDataId) {
			for(int tmpfId: curX.get(tmpDataId).keySet() ) 
				if( tmpfId > 0 ) {
					curX.get(tmpDataId).put(tmpfId, curX.get(tmpDataId).get(tmpfId) 
												+ stepLength*deltaX.get(tmpDataId).get(tmpfId) );
					System.out.println( ""+ tmpDataId + " " + tmpfId + "deltaX" + deltaX.get(tmpDataId).get(tmpfId) + " " + curX.get(tmpDataId).get(tmpfId) + " " + originX.get(tmpDataId).get(tmpfId));
				}
		}
		
		calcEffort();
	}
	
	private double getNormFeatureDiff(int dataId) {
		double l2normFeatureDiff = 0;
		for(int fId : curX.get(dataId).keySet()) {
			l2normFeatureDiff += Math.pow(curX.get(dataId).get(fId) - originX.get(dataId).get(fId), 2.0);
		}
		return Math.pow(l2normFeatureDiff, 0.5);
	}

	public void gradImplicitFunction()
	{
		double grad[][] = new double[problem.n][problem.n];
		double pred[] = new double[problem.l];
		for(int dataId = 0; dataId<problem.l; ++dataId)
			pred[dataId] = prediction(curX.get(dataId), learnedWeights);
		double l2norm = this.getWeightL2Norm(learnedWeights);
		
		for (int j = 0; j < problem.n; j++)
			for (int jj = 0; jj < problem.n; jj++) {
				grad[j][jj] = - Math.pow(learnedWeights.get(jj), 2.0) / Math.pow(l2norm, 3.0) ;
				if( j == jj )
					grad[j][jj] += 1.0 + 1.0/l2norm;
			}
		Matrix A = new Matrix(grad);
		Matrix invA = A.inverse();

		try {
			BufferedWriter fout = new BufferedWriter( new FileWriter( "InvGradWW", true));
	
			for(int k = 0; k < problem.n; k++) {
				for(int j = 0; j < problem.n; j++)
					fout.write(" "+invA.get(k, j));
				fout.write("\n");
			}
			fout.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		
		double gradFX[][] = new double[problem.n][(problem.n)*problem.l];
		
		Printer printer = new Printer();
		
		for(int k = 0; k < problem.n; k++) {
			for(int j = 0; j < problem.n; j++)
				for(int i = 0; i < problem.l; i++) {
					if( j==k && 1-labels.get(i)*pred[i]>0 ) { // if has nonzero hinge loss
						gradFX[k][j*problem.l + i] = -C*labels.get(i);
					//	System.out.println( "" + k + " " + (j*problem.l+i) + " " + gradFX[k][j*problem.l + i] );
					}
					else
						gradFX[k][j*problem.l + i] = 0.0;
				}
		}
		
		
		Matrix B = new Matrix(gradFX);
		
		gradImpFunc = invA.times(B);	
		gradImpFunc = gradImpFunc.times(-1.0);
		printer.initWriter("InvGradFW");
		printer.printMatrix( gradImpFunc );
		printer.closeWriter();
	}

	@Override
	public double getLossOriginX() {
		double loss = 0.0;
		for(int dataId = 0; dataId < problem.l; ++dataId)
			loss += Math.max(0, 1 - labels.get(dataId)*prediction( originX.get(dataId), learnedWeights ) );
		return loss;
	}

}
