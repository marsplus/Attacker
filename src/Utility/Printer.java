package Utility;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import AttackLearner.AttackLinearLearner;
import Jama.Matrix;


public class Printer {
	private BufferedWriter fout;
	
	/**
	 * initialize the writer to file
	 * @param fileName
	 */
	public void initWriter( String fileName ) {
		try {
			if( fout != null )
				fout.close();
			fout = new BufferedWriter(new FileWriter( fileName, true ));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void printFeatureModification( int label, AttackLinearLearner attacker ) {
		try {
			for(int i = 0; i < attacker.getNumDataInstances(); i++) 
			if( attacker.getLabel(i)==-1 ){
				for(int fId : attacker.getWeightIdSet() ) {
					if(fId > 0) fout.write(" ");
						fout.write(""+attacker.getFeatureDis(i, fId) );
						// TODO Auto-generated catch block
					}
				}
				fout.write("\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Print the response modification, with one feature value with index fId
	 * @param attacker   the attacker
	 * @param fId   the corresponding feature id
	 */
	public void printResponseModification( AttackLinearLearner attacker, int fId ) {
		try {
			for(int i = 0; i < attacker.getNumDataInstances(); i++) {
				fout.write(""+attacker.getCurFeature(i, fId) + " " + attacker.getOriginResponse(i) 
									+ " " + attacker.getCurResponse(i) + "\n" );
						// TODO Auto-generated catch block
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	public void printWeights(BufferedWriter foutP, AttackLinearLearner attacker) {
		try {
			foutP.write( "Fid" + 0 + "Weight " + attacker.getLearnedWeight(0) + "\n" );
			for(int fId : attacker.getWeightIdSet()) {			
				foutP.write( "Fid" + fId + "Weight " + attacker.getLearnedWeight(fId) + " " + "target Weight " + attacker.getTargetWeight(fId) + "\n" );
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void printMatrix( Matrix invA ) {
		try {	
			for(int j = 0; j < invA.getRowDimension(); j++) {
				for(int k = 0; k < invA.getColumnDimension(); k++)
					if( Math.abs(invA.get(j, k)) < 1e-3 )
						fout.write( "0 " );
					else 
						fout.write( invA.get(j,k) + " " );
				fout.write("\n");
			}
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}
	
	public void printLossEffort(BufferedWriter foutP, int iter, AttackLinearLearner attacker) {
		try {
			foutP.write( "" + (iter+1) + " " + attacker.getL1Effort() + " " + attacker.getL2Effort() + " " + attacker.weightDis(2) + " " + attacker.getLossOriginX() + "\n");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void printData(AttackLinearLearner attacker, boolean ifOrigin) {
		try {
			for(int i = 0; i < attacker.getNumDataInstances(); ++i) {
				int iter = 0;
				for(int fId : attacker.getWeightIdSet() ) 
					if( fId > 0 ) {
						if(iter > 0) fout.write(" ");
						iter += 1;
						if( ifOrigin )
							fout.write(""+attacker.getOriginFeature(i, fId) );
						else
							fout.write(""+attacker.getCurFeature(i, fId) );
					}
				fout.write(" " + attacker.getLabel(i) + "\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void closeWriter() {
		try {
			if( fout != null )
				fout.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
