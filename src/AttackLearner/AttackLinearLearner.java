package AttackLearner;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import Utility.Utility;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Train;



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
	
	// the mean feature values among data points
	HashMap<Integer, Double> meanX;

	// the L1 norm of feature values among data points
	HashMap<Integer, Double> normX;

	/**
	 * initialize all members
	 */
	AttackLinearLearner(){
		targetWeights = new HashMap<Integer, Double>();
				
		originX = new ArrayList<HashMap<Integer,Double> >();
		curX = new ArrayList<HashMap<Integer, Double> >();
		originY = new ArrayList<Double>();
		curY = new ArrayList<Double>();
			
		labels = new ArrayList<Integer>();
		
		L2Effort = 0.0;
		L1Effort = 0.0;
	}
	
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
	 * return current response of instance dataId
	 * @param i
	 * @return
	 */
	public double getCurResponse(int dataId) {
		return curY.get(dataId);
	}
	
	/**
	 * return original response of instance dataId
	 * @param i
	 * @return
	 */
	public double getOriginResponse(int dataId) {
		return originY.get(dataId);
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
	 * normalize each feature among all datapoints
	 * and scale the target weights accordingly		
	 * @param
	 * @return 
	 * @return
	 */
	public void normalizeDataAndTargetWeights() {
		meanX = new HashMap<Integer, Double>();
		normX = new HashMap<Integer, Double>();

		for(int fId : targetWeights.keySet()) {
			meanX.put(fId, 0.0);
			normX.put(fId, 0.0);
			
			for(int dataId=0; dataId<curX.size(); ++dataId)
				meanX.put( fId, meanX.get(fId)+curX.get(dataId).get(fId) );
			meanX.put(fId, meanX.get(fId)/curX.size());
			
			for(int dataId = 0; dataId < curX.size(); ++dataId) {
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)-meanX.get(fId));
				normX.put(fId, normX.get(fId) + Math.abs(curX.get(dataId).get(fId)) );
			}
			normX.put(fId, normX.get(fId)/curX.size());
			for(int dataId = 0; dataId < curX.size(); ++dataId) {
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)/normX.get(fId));
				//System.out.println(curX.get(dataId).get(fId));
			}
			
			targetWeights.put(fId, targetWeights.get(fId)*normX.get(fId));
		}
		
	}

	/**
	 * recover the dataset by times the norms and add the mean. 
	 * and scale the target weights accordingly		
	 * @param
	 * @return
	 */
	public void recoverDatasetFromMeanAndNorm() {
		// recover each feature 
		for(int fId : targetWeights.keySet()) {
			for(int dataId = 0; dataId < curX.size(); ++dataId)
				curX.get(dataId).put(fId, curX.get(dataId).get(fId)*normX.get(fId) + meanX.get(fId));
			learnedWeights.put(0, learnedWeights.get(0) - learnedWeights.get(fId)*meanX.get(fId)/normX.get(fId) );
			learnedWeights.put(fId, learnedWeights.get(fId)/normX.get(fId) );
			targetWeights.put(fId, targetWeights.get(fId)/normX.get(fId));
		}
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
	public double getFeatureDis(int i, int fId) {
		return originX.get(i).get(fId) - curX.get(i).get(fId);
	}
	
	/**
	 * get L2-norm of feature distance for dataId
	 * @param dataId
	 * @return L2-norm of feature distance
	 */
	private double getNormFeatureDiff(int dataId) {
		double l2normFeatureDiff = 0;
		for(int fId : curX.get(dataId).keySet()) {
			l2normFeatureDiff += Math.pow(curX.get(dataId).get(fId) - originX.get(dataId).get(fId), 2.0);
		}
		return Math.pow(l2normFeatureDiff, 0.5);
	}
}
