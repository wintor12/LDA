package lda;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

public class Vocabulary {
	public Map<Integer, String> idToWord = null;
	public Map<String, Integer> wordToId = null;
	public Map<String, Integer> wordCount = null;
	
	
	public Vocabulary()
	{
		idToWord = new TreeMap<Integer, String>();
		wordToId = new TreeMap<String, Integer>();
		wordCount = new TreeMap<String, Integer>();
		
	}
	
	public int size()
	{
		return wordCount.size();
	}
	
	public void getVocabulary(String path)
	{
		List<String> dir = Tools.listDir(path);
    	for(String d : dir)
    	{
    		String text = "";
    		try {
    			text = FileUtils.readFileToString(new File(path + d));
    		} catch (IOException e) {
    			e.printStackTrace();
    		}

    		String[] words = text.split(" ");
    		for(String word: words)
    		{
    			if(wordCount.containsKey(word))
    			{
    				wordCount.put(word, wordCount.get(word) + 1);
    			}
    			else
    			{
    				wordToId.put(word, this.size());
    				idToWord.put(this.size(), word);
    				wordCount.put(word, 1);	
    			}
    		}

    	}
    	printToFile(path + "../res/" + "idAndWord");
	}
	
	public void getAdj(String path)
	{
		
	}
	
	public void printToFile(String filepath) {
		StringBuilder sb = new StringBuilder();
		for (Map.Entry<Integer, String> entry : idToWord.entrySet())
		{
			int id = entry.getKey();
			String word = entry.getValue();
			sb.append(id + ":" + word);
			sb.append(System.getProperty("line.separator"));
		}
		try {
			FileUtils.writeStringToFile(new File(filepath), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		sb = new StringBuilder();
		for (Map.Entry<String, Integer> entry : wordCount.entrySet())
		{
			int count = entry.getValue();
			String word = entry.getKey();
			sb.append(word + ":" + count);
			sb.append(System.getProperty("line.separator"));
		}
		try {
			FileUtils.writeStringToFile(new File(filepath + "2"), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
