package gtrf;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
    Vocabulary voc;
    public Map<Integer, Integer> wordCount = null;
    public Map<Integer, Integer> idToIndex = null;
    public Map<Integer, List<Integer>> adj = null;
    
    double[] gamma;  //variational dirichlet parameter, K dimension  initialized when run EM
    double[][] phi; //variational multinomial, corpus.maxLength() * K dimension
    
    double zeta1; //Taylor approx
    double zeta2;
    double exp_ec;  //expectation of coherent edges
    int num_e;   //total number of edges
    double exp_theta_square; 
    
    int[] word_topic;//topic assigment to each word
    double[] theta; //topic assigment to this doc
    
    public Document(String path, String doc_name, Vocabulary voc)
    {
    	this.voc = voc;
    	this.path = path;
    	this.doc_name = doc_name;
    	wordCount = new TreeMap<Integer, Integer>();
    	idToIndex = new TreeMap<Integer, Integer>();
    	adj = new TreeMap<Integer, List<Integer>>();
    	try {
			this.text = FileUtils.readFileToString(new File(path + "data_words/" + doc_name));
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    //get adjacent words for each word from data_edges folder
    public void getEdges2()
    {
    	String text = "";
		try {
			text = FileUtils.readFileToString(new File(path + "data_edges_2/" + doc_name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(String line: text.split("\n"))
		{
			String word = line.substring(0, line.indexOf(':'));
			List<String> list = Arrays.asList(line.substring(line.indexOf('[') + 1, line.indexOf(']')).split(","));
			num_e += list.size();
			int wordid = voc.wordToId.get(word);
			List<Integer> adjList = new ArrayList<Integer>();
			for(String w: list)
			{
				adjList.add(voc.wordToId.get(w.trim()));
			}
			adj.put(wordid, adjList);
		}
		num_e = num_e/2; //we count num_e twice;
    }
    
    //format to  word: count and initialize each doc object
    //set word count map, set words, ids, counts array
    public void formatDocument() 
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
			idToIndex.put(id, i);
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
