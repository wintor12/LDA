package process;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

public class Edges_2 {
	
	static String path_words = "/Users/tongwang/Desktop/LDA/code/data_words4/data_words/";
	static String path_trees = "/Users/tongwang/Desktop/LDA/code/data_words4/data_trees/";
	static String path_edges_2 = "/Users/tongwang/Desktop/LDA/code/data_words4/data_edges_2/";
    Map<String, List<String>> map = new TreeMap<String, List<String>>();
	
	public static String extractParent(String wordNode)
	{
		return wordNode.substring(1, wordNode.indexOf('-'));
	}
	
	public static String extractWord(String wordNode)
	{
		return wordNode.substring(wordNode.indexOf(',') + 2, wordNode.indexOf('-', wordNode.indexOf(',')));
	}
	
	//get adjacency linklist for each word that is in bag of words
	//new File(path_trees + name) is the dependency tree for a document
	//new File(path_words + name) is the bag of words that removes stopwords and non letters 
	public void extractAndSave2Edges(String name)
	{
		String worddict = "";
		try {
			worddict = FileUtils.readFileToString(new File(path_words + name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		Set<String> dict = new HashSet<String>();   //word dictionary of current document
		for(String w: worddict.split(" "))
		{
			dict.add(w);
		}
		String text = "";
		try {
			text = FileUtils.readFileToString(new File(path_trees + name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder sb = new StringBuilder();
		String[] sentences = text.split("\n");
		for(String sent: sentences)
		{
			String[] wordNodes = sent.split("\t");
			for(String wordNode: wordNodes)
			{
				String word = extractWord(wordNode).toLowerCase();
				String parent = extractParent(wordNode).toLowerCase();
				//If words are not in dict, remove it
				if(parent.equals("ROOT") || !dict.contains(parent) || !dict.contains(word))
					continue;
				if(!map.containsKey(word))
					map.put(word, new ArrayList<String>());
				if(!map.containsKey(parent))
					map.put(parent, new ArrayList<String>());
				if(!map.get(parent).contains(word))
					map.get(parent).add(word);
				if(!map.get(word).contains(parent))
					map.get(word).add(parent);			
			}
		}
		for (Map.Entry<String, List<String>> entry : map.entrySet())
		{
			String term = entry.getKey();
			List<String> adj_term = entry.getValue();
			sb.append(term + ":" + adj_term.toString());
			sb.append(System.getProperty("line.separator"));
		}
		try {
			FileUtils.writeStringToFile(new File(path_edges_2 + name), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

	public static void main(String[] args) {
		for(String name: Process.listDir(path_trees))
		{
			Edges_2 ed = new Edges_2();
			ed.extractAndSave2Edges(name);
		}
//		String text = "";
//		try {
//			text = FileUtils.readFileToString(new File(path_edges_2 + "1"));
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//		String line = text.split("\n")[0];
////		for(String line: text.split("\n"))
//		{
//			String word = line.substring(0, line.indexOf(':'));
//			List<String> list = Arrays.asList(line.substring(line.indexOf('[') + 1, line.indexOf(']')).split(","));
//			System.out.println(word + ":" + list.toString());
//			for(String w : list)
//				System.out.println(w.trim());
//		}
		
	}

}
