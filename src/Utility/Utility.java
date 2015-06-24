package Utility;

public class Utility {
	public static double softShrinkage(double newValue, Double x0,
			double reg) {
		// TODO Auto-generated method stub
		if( newValue > x0 + reg )
			return newValue - reg;
		else if( newValue < x0 - reg )
			return newValue + reg;
		else return x0;
		
	}
	
	public static double softThreshold(double a, Double b, double threshold) {
		// TODO Auto-generated method stub
		if( a-b > threshold )
			return a-threshold;
		else if( a-b < -threshold )
			return a+threshold;
		else
			return b;
	}
	
	public static double sigmoid( double x ) {
		return 1.0 / (1 + Math.exp(-x));
	}
	
	public static int getLabel( double y )
	{
		if( Math.abs( y - 1 ) < 1e-7  )
			return 1;
		else
			return -1;
	}
}
