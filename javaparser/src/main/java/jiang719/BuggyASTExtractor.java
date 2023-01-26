package jiang719;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;


public class BuggyASTExtractor {
	
	public String outputDir = "";
	
	public BuggyASTExtractor(String outputDir) {
		this.outputDir = outputDir;
	}
	
	public String readLines(String filePath, int start, int end) throws IOException {
		String content = "";
		BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		int lineNum = 0;
		while ((line = br.readLine()) != null) {
			line += "\n";
			lineNum += 1;
			if (start <= lineNum && lineNum < end) {
				content += line;
			}
			if (lineNum == end)
				break;
		}
		br.close();
		return content.trim();
	}
	
	public void replaceRemByAdd(String remFilePath, int start, int end, String addContent) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(remFilePath)));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(this.outputDir + "tmp_add.java")));
		String line = "";
		int lineNum = 0;
		while ((line = br.readLine()) != null) {
			lineNum += 1;
			line += "\n";
			if (lineNum == start)
				bw.write(addContent + "\n");
			if (start <= lineNum && lineNum < end)
				continue;
			bw.write(line);
		}
		br.close();
		bw.close();
	}
	
	@SuppressWarnings({ "unchecked", "unused" })
	public void prepareData(String remFilePath, String addFilePath, int remStart, int remEnd, int addStart, int addEnd) throws IOException {
		String remLines = readLines(remFilePath, remStart, remEnd);
		String addLines = readLines(addFilePath, addStart, addEnd);
		
		replaceRemByAdd(remFilePath, remStart, remEnd, addLines);
		
		Node remContext = MyParser.findContextNode(remFilePath, remStart, remEnd);
		Node addContext = MyParser.findContextNode(this.outputDir + "tmp_add.java", remStart, remStart + addEnd - addStart);
		
		ArrayList<Node> coveredRemNodes = MyParser.findCoveredBuggyNode(remContext, remStart, remEnd - 1, remLines);
		Node remNode = null;
		if (coveredRemNodes.size() == 0)
			remNode = MyParser.findCoveringBuggyNode(remContext, remLines, remStart, remEnd);
		else
			remNode = MyParser.findCoveringBuggyNode(remContext, coveredRemNodes);
		
		ArrayList<String> contextBeforeAfter = MyParser.findContextBeforeAfter(
				remFilePath, remContext,
				remNode.getRange().get().begin.line, remNode.getRange().get().begin.column, 
				remNode.getRange().get().end.line, remNode.getRange().get().end.column
		);
		
		ArrayList<Integer> contextRange = MyParser.findContextRange(
				this.outputDir + "tmp_add.java", addContext, contextBeforeAfter.get(0), contextBeforeAfter.get(1)
		);
		int beforeLine = contextRange.get(0), beforeColumn = contextRange.get(1) + 1;
		int afterLine = contextRange.get(2), afterColumn = contextRange.get(3) - 1;
		if (contextBeforeAfter.get(0).equals("")) {
			beforeLine = addContext.getRange().get().begin.line;
			beforeColumn = addContext.getRange().get().begin.column;
		}
		if (contextBeforeAfter.get(1).equals("")) {
			afterLine = addContext.getRange().get().end.line;
			afterColumn = addContext.getRange().get().end.column;
		}
		
		ArrayList<Node> coveredAddNodes = MyParser.findCoveredBuggyNode(addContext, beforeLine, beforeColumn, afterLine, afterColumn);
		Collections.sort(coveredAddNodes, new Comparator<Node>() {
			 
            public int compare(Node n1, Node n2) {
                Range r1 = n1.getRange().get();
                Range r2 = n2.getRange().get();
                if (r1.begin.line == r2.begin.line) {
                	if (r1.begin.column == r2.begin.column) {
                		if (r1.end.line == r2.end.line)
                			return r1.end.column - r2.end.column;
                		return r1.end.line - r2.end.line;
                	}
                	return r1.begin.column - r2.begin.column;
                }
                return r1.begin.line - r2.begin.line;
            }
        });
		
		HashMap<Node, Integer> remIndices = MyParser.findBuggyNodeIndex(remContext, new ArrayList<Node>(Arrays.asList(remNode)));
		HashMap<Node, Integer> addIndices = MyParser.findBuggyNodeIndex(addContext, coveredAddNodes);
		
		HashMap<String, Object> result = new HashMap<String, Object>();
		if (remNode != null ) {
			Range range = remNode.getRange().get();
			result.put("rem_range", range.begin.line + "," + range.begin.column + "-" + range.end.line + "," + range.end.column);
			result.put("rem_index", remIndices.get(remNode).toString());
			result.put("rem_code", remNode.toString());
		}
		else {
			result.put("rem_range", "");
			result.put("rem_index", "");
			result.put("rem_code", "");
		}
		result.put("add_index", new ArrayList<String>());
		result.put("add_code", new ArrayList<String>());
		for (Node addNode : coveredAddNodes) {
			String index = "", code = "";
			if (addNode != null) {
				index = addIndices.get(addNode).toString();
				code = addNode.toString();
			}
			((ArrayList<String>) result.get("add_index")).add(index);
			((ArrayList<String>) result.get("add_code")).add(code);
		}
		
		result.put("rem_context", remContext == null ? "" : remContext.toString());
		result.put("add_context", addContext == null ? "" : addContext.toString());
		
		ObjectMapper writer = new ObjectMapper();
		writer.writeValue(new File(this.outputDir + "tmp.json"), result);
	}
	
	public void prepareRem(String remFilePath, int remStart, int remEnd) throws Exception {
		String remLines = readLines(remFilePath, remStart, remEnd);
		Node remContext = MyParser.findContextNode(remFilePath, remStart, remEnd);
		
		ArrayList<Node> coveredRemNodes = MyParser.findCoveredBuggyNode(remContext, remStart, remEnd - 1, remLines);
		Node remNode = null;
		if (coveredRemNodes.size() == 0)
			remNode = MyParser.findCoveringBuggyNode(remContext, remLines, remStart, remEnd);
		else
			remNode = MyParser.findCoveringBuggyNode(remContext, coveredRemNodes);
		
		HashMap<Node, Integer> remIndices = MyParser.findBuggyNodeIndex(remContext, new ArrayList<Node>(Arrays.asList(remNode)));
		
		HashMap<String, Object> result = new HashMap<String, Object>();
		if (remNode != null ) {
			Range range = remNode.getRange().get();
			result.put("rem_range", range.begin.line + "," + range.begin.column + "-" + range.end.line + "," + range.end.column);
			result.put("rem_index", remIndices.get(remNode).toString());
			result.put("rem_code", remNode.toString());
		}
		else {
			result.put("rem_range", "");
			result.put("rem_index", "");
			result.put("rem_code", "");
		}
		result.put("rem_context", remContext == null ? "" : remContext.toString());
		ObjectMapper writer = new ObjectMapper();
		writer.writeValue(new File(this.outputDir + "tmp.json"), result);
	}
	
	
	public static void main(String[] args) throws Exception {
		
		if (args[0].equals("defects4j") || args[0].equals("training")) {
			if (args.length == 8) {
				String outputDir = args[1];
				String remFilePath = args[2];
				String addFilePath = args[3]; 
				int remStart = Integer.parseInt(args[4]);
				int remEnd = Integer.parseInt(args[5]);
				int addStart = Integer.parseInt(args[6]);
				int addEnd = Integer.parseInt(args[7]);
				
				BuggyASTExtractor analyzer = new BuggyASTExtractor(outputDir);
				analyzer.prepareData(remFilePath, addFilePath, remStart, remEnd, addStart, addEnd);
			} else if (args.length == 5) {
				String outputDir = args[1];
				String remFilePath = args[2];
				int remStart = Integer.parseInt(args[3]);
				int remEnd = Integer.parseInt(args[4]);
				
				BuggyASTExtractor analyzer = new BuggyASTExtractor(outputDir);
				analyzer.prepareRem(remFilePath, remStart, remEnd);
			}
		}
		
	}

}
