import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.*;
import java.util.Properties;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;


/** This is a very simple demo of calling the Chinese Word Segmenter
 *  programmatically.  It assumes an input file in UTF8.
 *  <p/>
 *  <code>
 *  Usage: java -mx1g -cp seg.jar SegDemo fileName
 *  </code>
 *  This will run correctly in the distribution home directory.  To
 *  run in general, the properties for where to find dictionaries or
 *  normalizations have to be set.
 *
 *  @author Christopher Manning
 */

public class SegData {

  private static final String basedir = System.getProperty("SegDemo", "data");
  private static Map<String, Integer> M = new HashMap<String, Integer>();
  
  public static ArrayList<ArrayList<String>> readInSentences(String Filename) {
	  ArrayList<ArrayList<String>> ret = new ArrayList<ArrayList<String> >();
	  try {
		BufferedReader br = new BufferedReader(new FileReader(Filename));

		String str = "";
		Pattern pt = Pattern.compile("[0-9]*");
		Boolean isFirst = true;
		
		ArrayList<String> tmp = new ArrayList<String>();
		
		while ((str=br.readLine()) != null) {
			Matcher isNum = pt.matcher(str);
			if (!isNum.matches()) {
				tmp.add(str);
			} else {
				if (isFirst) {
					isFirst = false;
				} else {
					ret.add(new ArrayList<String> (tmp));
					tmp.clear();
				}
			}
		}
		
		ret.add(tmp);
		
		br.close();

	} catch (FileNotFoundException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
	  
	  return ret;
  }
  
  public static ArrayList<ArrayList<List<Integer>>> doMap(ArrayList<ArrayList<String> > sts, CRFClassifier<CoreLabel> segmenter) {
	  ArrayList<ArrayList<List<Integer>>> ret = new ArrayList<ArrayList<List<Integer>>>();
	  
	  // add flag for start of sentence and end of sentence
	  
	  M.put("<S>", 1);
	  M.put("</S>", 2);
	  
	  int dicCnt = 3;
	  
	  for (int i = 0; i < sts.size(); ++i) {
		  ArrayList<String> cur = sts.get(i);
		  for (int j = 0; j < cur.size(); ++j) {
			  //System.out.println(cur.get(j));
			  List<String> tmp = segmenter.segmentString(cur.get(j));
			  //System.out.println(tmp);
			  for (String e : tmp) {
				  if (M.containsKey(e)) {
					  continue;
				  } else {
					  M.put(e, dicCnt);
					  dicCnt++;
				  }
			  }
		  }
	  }
	  
	  for (int i = 0; i < sts.size(); ++i) {
		  ArrayList<String> cur = sts.get(i);
		  ArrayList<List<Integer>> _ = new ArrayList<List<Integer>>();
		  for (int j = 0; j < cur.size(); ++j) {
			  List<String> tmp = segmenter.segmentString(cur.get(j));
			  List<Integer> __ = new ArrayList<Integer>();
			  for (String e : tmp) {
				  __.add(M.get(e));
			  }
			  _.add(__);
		  }
		  ret.add(_);
	  }
	  
	  return ret;
  }
  
  public static void outputVector(ArrayList<ArrayList<List<Integer>>> res, String fileName) {
	  try {
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		
		Integer cnt = 1;
		
		for (ArrayList<List<Integer>> e : res) {
			bw.write(cnt.toString());
			bw.write("\n");
			for (List<Integer> f : e) {
				// start
				bw.write("1 ");
				for (Integer g : f) {
					bw.write(g.toString());
					bw.write(' ');
				}
				// end
				bw.write("2");
				bw.write('\n');
			}
			cnt += 1;
		}
		
		bw.close();
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }
  
  public static void outputMap(String fileName) {
	  try {
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		
		Iterator it = M.entrySet().iterator();
		
		while (it.hasNext()) {
			Map.Entry pair = (Map.Entry)it.next();
			bw.write(pair.getKey() + " " + pair.getValue().toString() + "\n");
			it.remove();
		}
		
		bw.close();
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }
  
  public static void main(String[] args) throws Exception {
    //System.setOut(new PrintStream(System.out, true, "utf-8"));

    Properties props = new Properties();
    props.setProperty("sighanCorporaDict", basedir);
    // props.setProperty("NormalizationTable", "data/norm.simp.utf8");
    // props.setProperty("normTableEncoding", "UTF-8");
    // below is needed because CTBSegDocumentIteratorFactory accesses it
    props.setProperty("serDictionary", basedir + "/dict-chris6.ser.gz");
    if (args.length > 0) {
      props.setProperty("testFile", args[0]);
    }
    props.setProperty("inputEncoding", "UTF-8");
    props.setProperty("sighanPostProcessing", "true");

    CRFClassifier<CoreLabel> segmenter = new CRFClassifier<>(props);
    segmenter.loadClassifierNoExceptions(basedir + "/ctb.gz", props);
    
    String inSentencesFileName = "D:\\train.txt";
    String outSentencesFileName = "D:\\train_vector.txt";
    String outDictionaryFileName = "D:\\dictionary.txt";
    
    outputVector(doMap(readInSentences(inSentencesFileName), segmenter), outSentencesFileName);
    
    outputMap(outDictionaryFileName);
  }

}
