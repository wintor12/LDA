package lda;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

public class Document {	
	String text;  //the content of the document, contains words from Python tokenizer.
	String path;
	String[] words;
    int[] counts;
    int[] ids;
    int length;  //Total unique words
    int total;   //Total words
    String doc_name;
    public Map<Integer, Integer> wordCount = null;
    
    int[] word_topic;
    double[] theta;
    
    double[] gamma;  //variational dirichlet parameter, K dimension  initialized when run EM
    double[][] phi; //variational multinomial, corpus.maxLength() * K dimension
    
    public Document(String path, String doc_name)
    {
    	this.path = path;
    	this.doc_name = doc_name;
    	wordCount = new TreeMap<Integer, Integer>();
    	try {
			this.text = FileUtils.readFileToString(new File(path + "data_words/" + doc_name));
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    //format to  word: count and initialize each doc object
    //set word count map, set words, ids, counts array
    public void formatDocument(Vocabulary voc) 
    {
    	String[] ws = text.split(" ");
    	for(String word: ws)  //put word count pair to map
    	{
    		int id = voc.wordToId.get(word);
    		if(!wordCount.containsKey(id))
    		{
    			wordCount.put(id, 1);
    		}
    		else
    		{
    			wordCount.put(id, wordCount.get(id) + 1);
    		}
    	}
    	this.length = wordCount.size();
    	words = new String[wordCount.size()];
    	counts = new int[wordCount.size()];
    	ids = new int[wordCount.size()];
    	int i = 0;
    	for (Map.Entry<Integer, Integer> entry : wordCount.entrySet())
		{
    		
			int id = entry.getKey();
			int count = entry.getValue();
			words[i] = voc.idToWord.get(id);
			counts[i] = count;
			ids[i] = id;
			i++;
			this.total += count;
		}
    }
    
    //Assign topic to each word based on beta
    //Compute theta according to the counts of each topic
    public void assign_topic_to_word(Model model)
    {
    	word_topic = new int[ids.length];
    	theta = new double[model.num_topics];
    	for(int n = 0; n < word_topic.length; n++)
    	{
    		double max = model.log_prob_w[0][ids[n]];
    		int max_k = 0;
    		for(int k = 1; k < model.num_topics; k++)
    		{
				if(model.log_prob_w[k][ids[n]] > max)
				{
					max = model.log_prob_w[k][ids[n]];
					max_k = k;
				}
    		}
    		word_topic[n] = max_k;
    		theta[max_k] += counts[n];
    	}
    	 //Compute theta for a document
    	for(int k = 0; k < model.num_topics; k++)
    	{
    		theta[k] = (double)(theta[k] + model.alpha)/(total + model.num_topics * model.alpha);
    	}
    }

}
