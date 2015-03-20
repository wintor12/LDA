package lda;

import org.apache.commons.math3.random.MersenneTwister;

public class Suffstats {
	double[][] class_word;
	double[] class_total;
	double alpha_suffstas;
	int num_topics;
	int num_terms;
	int num_docs;
	
	public Suffstats(Model model)
	{
		num_topics = model.num_topics;
		num_terms = model.num_terms;
		class_total = new double[num_topics];
		class_word = new double[num_topics][num_terms];
	}
	
	//Random initialize word-topic joint probability
	public void random_initialize_ss()
	{
		MersenneTwister mt = new MersenneTwister(32);
		int k, n;
		for (k = 0; k < num_topics; k++)
	    {
	        for (n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] += 1.0/num_terms + mt.nextDouble();
	            class_total[k] += class_word[k][n];
	        }
	    }
	}
	
	public void zero_initialize_ss()
	{
		int k, n;
		for (k = 0; k < num_topics; k++)
	    {
			class_total[k] = 0;
	        for (n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] = 0;	            
	        }
	    }
		this.num_docs = 0;
		this.alpha_suffstas = 0;
	}
}
