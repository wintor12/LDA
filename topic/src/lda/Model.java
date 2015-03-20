package lda;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.special.Gamma;

public class Model {
	
	double[][] log_prob_w; //log beta
	double alpha;
	int num_topics;
	int num_terms;
	static double NEWTON_THRESH = 1e-5;
	static int MAX_ALPHA_ITER = 1000;
	
	public Model(int num_topics, int num_terms, double alpha)
	{
		this.num_terms = num_terms;
		this.num_topics = num_topics;
		log_prob_w = new double[num_topics][num_terms]; 
		this.alpha = alpha;
	}
	
	//if init is true, then we use the initial alpha,
	//else update alpha
	public void mle(Suffstats ss, boolean init)
	{
		int k, w;
		//Update beta from sufficient statistics
		for (k = 0; k < num_topics; k++)
	    {
	        for (w = 0; w < num_terms; w++)
	        {	        	
	        	//log p(w|k) = log(p(w,k)/p(k)) = log p(w,k) - log p(w)
	            if (ss.class_word[k][w] > 0)
	                this.log_prob_w[k][w] = Math.log(ss.class_word[k][w]) - Math.log(ss.class_total[k]);
	            else
	                this.log_prob_w[k][w] = -100;
	        }
	    }
		
		//Update alpha
		if(init == false)
			this.alpha = opt_alpha(ss.alpha_suffstas, ss.num_docs, this.num_topics);
	}
	
	/*
	 * newtons method
	 * @D ss.num_topics
	 * @K topics
	 */
	double opt_alpha(double ss, int D, int K)
	{
	    double a, log_a, init_a = 100;
	    double f, df, d2f;
	    int iter = 0;

	    log_a = Math.log(init_a);
	    do
	    {
	        iter++;
	        a = Math.exp(log_a);
	        if (Double.isNaN(a))
	        {
	            init_a = init_a * 10;
	            System.out.println("warning : alpha is nan; new init = " +  init_a);
	            a = init_a;
	            log_a = Math.log(a);
	        }
	        f = D * (Gamma.logGamma(K * a) - K * Gamma.logGamma(a)) + (a - 1) * ss;
	        df = D * (K * Gamma.digamma(K * a) - K * Gamma.digamma(a)) + ss;
	        d2f = D * (K * K * Gamma.trigamma(K * a) - K * Gamma.trigamma(a));
	        log_a = log_a - df/(d2f * a + df);
//	        System.out.format("alpha maximization : %5.5f   %5.5f\n", f, df);
	    }
	    while ((Math.abs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
	    return(Math.exp(log_a));
	}
	
	public void save_lda_model(String path, String filename)
	{
		DecimalFormat df = new DecimalFormat("#.##");
		String name = path + filename + "_beta";
		StringBuilder sb = new StringBuilder();
		int k, w;
		for (k = 0; k < num_topics; k++)
	    {
	        for (w = 0; w < num_terms; w++)
	        {
	        	sb.append(df.format(this.log_prob_w[k][w]) + " ");
	        }
	        sb.append(System.getProperty("line.separator"));
        }
		try {
			FileUtils.writeStringToFile(new File(name), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String name2 = path + filename + "_other";
		StringBuilder sb2 = new StringBuilder();
		sb2.append("num_topics: " + this.num_topics + "\n");
		sb2.append("num_terms: " + this.num_terms + "\n");
		sb2.append("alpha: " + df.format(this.alpha) + "\n");
		try {
			FileUtils.writeStringToFile(new File(name2), sb2.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
