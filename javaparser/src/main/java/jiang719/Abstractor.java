package jiang719;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Set;

import edu.wm.cs.src2abs.lexer.JLexerTokenizer;
import edu.wm.cs.src2abs.parser.Parser;
import edu.wm.cs.src2abs.vocabulary.IdiomManager;

public class Abstractor {
	
	String idiomsFilePath = "idioms.txt";
	String mapOutputPath = "map.txt";
	
	public Abstractor() {
		
	}
	
	public Abstractor(String idiomsFilePath, String mapOutputPath) {
		this.idiomsFilePath = idiomsFilePath;
		this.mapOutputPath = mapOutputPath;
	}

	public String abstractCode(String code) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter("tmp.txt"));
		bw.write(code);
		bw.close();
		
		Set<String> idioms = IdiomManager.readIdioms(this.idiomsFilePath);
		
		Parser parser = new Parser(Parser.CodeGranularity.METHOD);
		parser.parseCode(code);
		
		JLexerTokenizer tokenizer = new JLexerTokenizer();
		tokenizer.setTypes(parser.getTypes());
		tokenizer.setMethods(parser.getMethods());
		tokenizer.setIdioms(idioms);

		String result = tokenizer.tokenize("tmp.txt").trim();
		tokenizer.exportMaps(this.mapOutputPath);
		return result;
	}
	
	public void getNoneIdentifiers() throws Exception {
		HashMap<String, Integer> vocabulary = new HashMap<String, Integer>();
		Path path = Paths.get("nonidentifier.txt");
		if (Files.exists(path)) {
			BufferedReader br = new BufferedReader(new FileReader("nonidentifier.txt"));
			String line = br.readLine();
			while (line != null) {
				line = line.trim();
				vocabulary.put(line.split(" ")[0], Integer.parseInt(line.split(" ")[1]));
				line = br.readLine();
			}
			br.close();
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter("nonidentifier.txt"));
		BufferedReader br = new BufferedReader(new FileReader("ctx_localize_combine.txt"));
		String line = br.readLine();
		int cnt = 0;
		while (line != null) {
			cnt += 1;
			if (cnt % 1000 == 0) {
				System.out.println(cnt + " " + vocabulary.size());
			}
			try {
				String code = line.split("\t")[0];
				String result = abstractCode(code);
				String[] tokens = result.split("\\s+");
				for (String token : tokens) {
					token = token.trim();
					if (token.length() >= 4 && token.substring(0, 4).equals("VAR_"))
						continue;
					if (token.length() >= 5 && token.substring(0, 5).equals("TYPE_"))
						continue;
					if (token.length() >= 7 && token.substring(0, 7).equals("METHOD_"))
						continue;
					if (token.length() >= 4 && token.substring(0, 4).equals("INT_"))
						continue;
					if (token.length() >= 6 && token.substring(0, 6).equals("FLOAT_"))
						continue;
					if (token.length() >= 7 && token.substring(0, 7).equals("STRING_"))
						continue;
					if (token.length() >= 5 && token.substring(0, 5).equals("CHAR_"))
						continue;
					if (! vocabulary.containsKey(token))
						vocabulary.put(token, 0);
					vocabulary.put(token, vocabulary.get(token) + 1);
				}
			}
			catch (Exception e) {
				
			}
			line = br.readLine();
		}
		System.out.println(vocabulary.size());
		for (String token : vocabulary.keySet()) {
			bw.write(token + " " + vocabulary.get(token) + "\n");
		}
		
		bw.close();
		br.close();
	}
	
	public void abstractLocalize(String filename, String outputname) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputname));
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = br.readLine();
		int cnt = 0, failed = 0;
		while (line != null) {
			line = line + " ";
			cnt += 1;
			if (cnt % 1000 == 0)
				System.out.println(cnt);
			
			for (int i = 0; i < 2; i += 1) {
				HashMap<String, String> mapping = new HashMap<String, String>();
				try {
					String code = line.split("\t")[i];
                    if (code.trim().length() > 0) {
                        abstractCode(code);
                        
                        BufferedReader reader = new BufferedReader(new FileReader("map.txt"));
                        String mapline = reader.readLine();
                        while (mapline != null) {
                            if (mapline.trim().equals("")) {
                                mapline = reader.readLine();
                                continue;
                            }
                            String[] names = mapline.trim().split("<MAP>");
                            mapline = reader.readLine();
                            String[] abs = mapline.trim().split("<MAP>");
                            assert(names.length == abs.length);
                            for (int j = 0; j < names.length; j += 1) {
                                if (names[j].equals(""))
                                    continue;
                                mapping.put(names[j], abs[j]);
                            }
                            mapline = reader.readLine();
                        }
                        reader.close();
                    }
				}
				catch (Exception e) {
					//e.printStackTrace();
					mapping = new HashMap<String, String>();
					failed += 1;
					System.out.println(cnt + " " + failed);
					//System.out.println(i);
					//System.out.println(line);
					// mapping.put("ERROR", "ERROR");
				}
				
				for (String name : mapping.keySet())
					bw.write(name + "<MAP>" + mapping.get(name) + " <SEP> ");
				if (i == 0)
					bw.write("\t");
				else
					bw.write("\n");
			}
			
			line = br.readLine();
		}
		bw.close();
		br.close();
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length == 2) {
            String input = args[0];
            String output = args[1];
			Abstractor abstractor = new Abstractor("data/idioms.txt", "map.txt");
			abstractor.abstractLocalize(input, output);
		}
	}

}
