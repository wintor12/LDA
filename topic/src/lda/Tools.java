package lda;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

public class Tools {
	/*
	 * list files in a folder
	 */	
	public static List<String> listDir(String folder)
	{
		File f = new File(folder);
		String fe[] = f.list();
		if(fe == null)
		{
			System.out.println("No such directory!!");
			return null;
		}
		List<String> files = new LinkedList<String>(Arrays.asList(fe));
		for(int i = 0; i < files.size(); i++)
		{
			if(files.get(i).equals(".DS_Store")||files.get(i).equals(".DS_S.txt")||
					new File(folder + files.get(i)).isDirectory())
				files.remove(i);
		}
		return files;
	}
	
	/*
	 * taylor approximation of first derivative of the log gamma function
	 *
	 */
	public static double digamma(double x)
	{
	    double p;
	    x=x+6;
	    p=1/(x*x);
	    p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p;
	    p=p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	    return p;
	}
	
	/*
	 * given log(a) and log(b), return log(a + b)
	 *
	 */
	static double log_sum(double log_a, double log_b) {
		double v;

		if (log_a < log_b) {
			v = log_b + Math.log(1 + Math.exp(log_a - log_b));
		} else {
			v = log_a + Math.log(1 + Math.exp(log_b - log_a));
		}
		return (v);
	}
	
	class ArrayIndexComparator implements Comparator<Integer>
	{
		double[] array;
		public ArrayIndexComparator(double[] array)
	    {
	        this.array = array;
	    }
		
		public Integer[] createIndexArray() 
		{
			Integer[] indexes = new Integer[array.length];
			for (int i = 0; i < array.length; i++) {
				indexes[i] = i; 
			}
			return indexes;
		}


		@Override
		public int compare(Integer index1, Integer index2) {
			if(array[index1] < array[index2])
				return 1;
			else if(array[index1] > array[index2])
				return -1;
			else
				return 0;
		}


	}
}
