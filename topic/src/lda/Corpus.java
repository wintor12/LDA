package lda;

import java.util.List;


public class Corpus {
	Document[] docs;
	Document[] docs_test;
	int num_terms;
    int num_docs;
    int num_docs_test;
    Vocabulary voc;
    
    //The files in this path are already tokenized and removed stop words in Python
    public Corpus(String path)
    {    	
    	// Iterate all files and get vocabulary, word id maps.
    	voc = new Vocabulary();
    	voc.getVocabulary(path + "data_words/");
    	num_terms = voc.size();
    	System.out.println("number of terms   :" + num_terms);
    	List<String> train = Tools.listDir(path + "train/");
    	num_docs = train.size();
    	System.out.println("number of training docs    :" + num_docs);
    	docs = new Document[num_docs];
    	int i = 0;
    	System.out.println("=======process training set========");
		for(String d : train)
		{
			if(i%10 == 0)
				System.out.println("Loading document " + i);
			Document doc = new Document(path, d);
			doc.formatDocument(voc); //format document to word: count, and set words, counts, ids array
//			System.out.println("Document " + d + " contain unique words : " + doc.length);
			docs[i] = doc;
			i++;
		}
		
		List<String> test = Tools.listDir(path + "test/");
		num_docs_test = test.size();
		System.out.println("number of test docs    :" + num_docs_test);
    	docs_test = new Document[num_docs_test];
		System.out.println("=======process test set========");	
		i = 0;
		for(String d : test)
		{
			if(i%10 == 0)
				System.out.println("Loading document " + i);
			Document doc = new Document(path, d);
			doc.formatDocument(voc); //format document to word: count, and set words, counts, ids array
//			System.out.println("Document " + d + " contain unique words : " + doc.length);
			docs_test[i] = doc;
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
