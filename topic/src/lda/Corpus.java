package lda;

import java.io.File;
import java.util.List;


public class Corpus {
	Document[] docs;
	int num_terms;
    int num_docs;    
    Vocabulary voc;
    
    //The files in this path are already tokenized and removed stop words in Python
    public Corpus(String path)
    {
    	List<String> dir = Tools.listDir(path + "data_words/");
    	// Iterate all files and get vocabulary, word id maps.
    	voc = new Vocabulary();
    	voc.getVocabulary(path + "data_words/");
    	num_terms = voc.size();
    	System.out.println("number of terms   :" + num_terms);
    	num_docs = dir.size();
    	System.out.println("number of docs    :" + num_docs);
    	docs = new Document[num_docs];
    	int i = 0;
		for(String d : dir)
		{
			Document doc = new Document(path, d);
			doc.formatDocument(voc); //format document to word: count, and set words, counts, ids array
			System.out.println("Document " + d + " contain unique words : " + doc.length);
			docs[i] = doc;
			i++;
		}
    }
    
    public int maxLength()
    {
    	int max = 0;
    	for(int i = 0; i < docs.length; i++)
    		max = max > docs[i].length?max:docs[i].length;
    	return max;
    }
    

}
