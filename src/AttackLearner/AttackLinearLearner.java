package AttackLearner;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import de.bwaldvogel.liblinear.InvalidInputDataException;



public abstract class AttackLinearLearner {
	// The target weights (used only in attractive attacks)
	protected HashMap<Integer, Double> targetWeights;
	
	// the currently learned weights (used only in attacking linear models)
	protected HashMap<Integer, Double> learnedWeights;
	
	// the original feature matrix
	protected ArrayList<HashMap<Integer, Double> > originX;
	
	// the current feature matrix after attacking
	protected ArrayList<HashMap<Integer, Double> > curX;

	// the current responses after attacking
	protected ArrayList<Double> curY;

	// the original responses
	protected ArrayList<Double> originY;

	// the label vector
	protected ArrayList<Integer> labels;
	
	// the L2-norm attack effort
	protected double L2Effort;
	
	// the L1-norm attack effort
	protected double L1Effort;
	
		
	
	/**
	 * return number of training data instance
	 */
	public int getNumDataInstances() {
		return originX.size();
	}
	
	/**
	 * return original feature value 
	 * @param dataId data Id
	 * @param fId feature Id
	 * @return
	 */
	public double getOriginFeature(int dataId, int fId) {
		return originX.get(dataId).get(fId);
	}

	/**
	 * return current feature value 
	 * @param dataId data Id
	 * @param fId feature Id
	 * @return
	 */
	public double getCurFeature(int dataId, int fId) {
		return curX.get(dataId).get(fId);
	}

	
	/**
	 * return label of instance dataId
	 * @param i
	 * @return
	 */
	public int getLabel(int dataId) {
		return labels.get(dataId);
	}
	
	/**
	 * get the L2-norm attack effort
	 * @return
	 */
	public double getL2Effort() {
		// TODO Auto-generated method stub
		return L2Effort;
	}

	/**
	 * get the L1-norm attack effort
	 * @return
	 */
	public double getL1Effort() {
		// TODO Auto-generated method stub
		return L1Effort;
	}

	/**
	 * get the key set of learnedWeights
	 * @return
	 */
	public Set<Integer> getWeightIdSet() {
		return targetWeights.keySet();
	}
	
	/**
	 * learned weight at fId
	 * @param fId
	 * @return
	 */
	public double getLearnedWeight(int fId) {
		return learnedWeights.get(fId);
	}
	
	/**
	 * target weight at fId
	 * @param fId
	 * @return
	 */
	public double getTargetWeight(int fId) {
		return targetWeights.get(fId);
	}
	
	/**
	 * get the distance between the learned weights and the target weights
	 * @param norm the norm of the distance
	 * @return
	 */
	public double weightDis(double norm)
	{
		double res = 0;
		for( int i : targetWeights.keySet()) 
			res += Math.pow(Math.abs(targetWeights.get(i) - learnedWeights.get(i)), norm);
		return res;
	}
		
	/**
	 * get the distance between the specific feature of the learned weights and the target weights
	 * @param targetFid the specific feature Id
	 * @param norm  the norm of the distance
	 * @return
	 */
	double weightDisTargetFeature(int targetFid, double norm)
	{
		double res = 0;
		res += Math.pow(Math.abs(targetWeights.get(targetFid) - learnedWeights.get(targetFid)), norm);
		return res;
	}
	
	/**
	 * get the difference of the specific feature of the learned weight and the target weights
	 * @param fId the specific feature Id
	 * @return
	 */
	double weightDis(int fId)
	{
		double res = learnedWeights.get(fId) - targetWeights.get(fId);
		return res;
	}
	
	/**
	 * calculate the L1 and L2 effort
	 */
	public void calcEffort() {
		L1Effort = 0;
		L2Effort = 0;
		for(int dataId = 0; dataId < curX.size(); dataId++)
			for(int fId : curX.get(dataId).keySet()) {
				L2Effort += Math.pow( curX.get(dataId).get(fId) - originX.get(dataId).get(fId) , 2.0);
				L1Effort += Math.abs( curX.get(dataId).get(fId) - originX.get(dataId).get(fId) );
			}
	}

	/**
	 * calculate the L1 and L2 effort on response
	 */
	public void calcYEffort() {
		L1Effort = 0;
		L2Effort = 0;
		for(int dataId = 0; dataId < curX.size(); dataId++) {
				L2Effort += Math.pow( curY.get(dataId) - originY.get(dataId) , 2.0);
				L1Effort += Math.abs( curY.get(dataId) - originY.get(dataId) );
			}
	}
	
	/**
	 * the loss of learned weight on dataId (only for debug)
	 * @param dataId data instance Id
	 * @return
	 */
	abstract double calcLoss(int dataId, HashMap<Integer, Double> weights);
		
	/**
	 * prediction of model for one data point
	 * @param x feature vectors of one data point
	 * @return
	 */
	public double prediction(HashMap<Integer, Double> x, HashMap<Integer, Double> weights) {
		double res = 0;
		for(int i : x.keySet()) {
			if( weights.containsKey(i) )
			res += x.get(i) * weights.get(i);
		}
		return res;
	}

	/**
	 * Load the data
	 * @param DATA_PATH  filePath for data
	 * @param targetWeightFileName  filePath for target weights
	 * @throws FileNotFoundException 
	 * @throws IOException 
	 * @throws NumberFormatException 
	 * @throws InvalidInputDataException 
	 */
	public abstract void initData( String DATA_PATH, String targetWeightFileName, int regNorm2) throws FileNotFoundException, NumberFormatException, IOException, InvalidInputDataException;
	
	/**
	 * learn the model   
	 * @param MODEL_PATH   Path to store the model
	 * @param MODEL_NAME   
	 * @throws IOException 
	 */
	public abstract void learnModel(String MODEL_PATH, String MODEL_NAME) throws IOException;
	
	/**
	 * Take the gradient (do attack) on feature matrix 
	 * @param dataId
	 * @param fId
	 * @param stepLength
	 * @param regularization
	 * @param fout
	 */
	public abstract void takeGradient(double stepLength, double regularization, BufferedWriter fout);

	/**
	 * return total loss w.r.t. originX
	 * @return
	 */
	public abstract double getLossOriginX();
	
	/**
	 * difference between original feature value and modified feature value
	 * @param i
	 * @param fId
	 * @return
	 */
	public double featureDis(int i, int fId) {
		return originX.get(i).get(fId) - curX.get(i).get(fId);
	}

}
