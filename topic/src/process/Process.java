package process;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;

import org.apache.commons.io.FileUtils;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

public class Process {
	
	static String path_documents = "/Users/tongwang/Desktop/LDA/code/data/";
	static String path_words = "/Users/tongwang/Desktop/LDA/code/data_words5/data_words/";
	static String path_trees = "/Users/tongwang/Desktop/LDA/code/data_words5/data_trees/";
	static String path_stopwords = "/Users/tongwang/Desktop/LDA/code/stopwords2.txt";
	static List<String> stopwords;
	
	public static void getStopwords()
	{
		String text = "";
		try {
			text = FileUtils.readFileToString(new File(path_stopwords));
		} catch (IOException e) {
			e.printStackTrace();
		}
		String[] words = text.split(" ");
		stopwords = Arrays.asList(words);
	}
	
	//Generate bag of words documents and dependency trees documents from "path_documents"
	//Output to path_words and path_trees
	public static void getWordsAndTrees()
	{		
		LexicalizedParser lp = LexicalizedParser.loadModel(
                "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
                "-maxLength", "80", "-retainTmpSubcategories");
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		for(String name: listDir(path_documents))
		{
			System.out.println(name);
			StringBuilder sb = new StringBuilder(); //bag of words
			StringBuilder sb2 = new StringBuilder(); //trees
			String text = "";
			try {
				text = FileUtils.readFileToString(new File(path_documents + name));
			} catch (IOException e) {
				e.printStackTrace();
			}
			Annotation document = new Annotation(text);			
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap s: sentences)
			{
				ArrayList<String> sent = new ArrayList<String>();
				for (CoreLabel token: s.get(TokensAnnotation.class)) {
					String word = token.toString().toLowerCase();
					if(word.matches("[a-z]+"))
					{
						if(!stopwords.contains(word))
							sb.append(word + " ");
						sent.add(token.get(TextAnnotation.class));
					}					
				}
				try {
					String[] sentence = sent.toArray(new String[sent.size()]);
					Tree parse = lp.apply(Sentence.toWordList(sentence)); 
					GrammaticalStructure gs = gsf.newGrammaticalStructure(parse); 
					Collection<TypedDependency> tdl = gs.typedDependenciesCCprocessed(); 
	//				System.out.println(tdl);
					Iterator it = tdl.iterator();
					while(it.hasNext())
					{
						String wordnode = it.next().toString();
						sb2.append(wordnode.substring(wordnode.indexOf('('), wordnode.indexOf(')') + 1) + "\t");
					}
					sb2.append("\n");
				} catch (Exception e) {
					System.out.println("error sentence:" + s.toString());
				}
			}
			try {
				FileUtils.writeStringToFile(new File(path_words + name), sb.toString());
				FileUtils.writeStringToFile(new File(path_trees + name), sb2.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
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

	public static void main(String[] args) {
		getStopwords();
		getWordsAndTrees();

	}

}
