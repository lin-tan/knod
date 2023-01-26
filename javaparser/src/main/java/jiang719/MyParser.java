package jiang719;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import com.github.javadocparser.TokenMgrError;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.Name;


public class MyParser {
	
	public static JavaParser parser = new JavaParser();
	
	public static Node parseMethodDeclaration(String code) {
		try {
			Node root = parser.parseMethodDeclaration(code).getResult().get();
			MyParser.removeComments(root);
			MyParser.removeAnnotation(root);
			return root;
		} catch (ParseProblemException e) {
			return null;
		} catch (Exception e) {
			return null;
		}
	}
	
	public static CompilationUnit parseFile(String filename) throws FileNotFoundException {
		try {
			Node root = parser.parse(new File(filename)).getResult().get();
			MyParser.removeComments(root);
			MyParser.removeAnnotation(root);
			return (CompilationUnit) root;
		} catch (ParseProblemException e) {
			return null;
		} catch (Exception e) {
			return null;
		}
	}
	
	public static CompilationUnit parseFile(File file) throws com.github.javadocparser.ParseException {
		try {
			Node root = parser.parse(file).getResult().get();
			MyParser.removeComments(root);
			MyParser.removeAnnotation(root);
			return (CompilationUnit) root;
		} catch (TokenMgrError e) {
			return null;
		} catch (Exception e) {
			return null;
		}
	}
	
	public static boolean containLines(Node node, int beginline, int endline, String subnodecode) {
		if (! node.getRange().isPresent())
			return false;
		Range noderange = node.getRange().get();
		if (noderange.begin.line > beginline || noderange.end.line < endline)
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String subnodestring = subnodecode.replaceAll("\\s+", "");
		return nodestring.contains(subnodestring);
	}
	
	public static boolean containLines(Node node, String subnodecode) {
		if (! node.getRange().isPresent())
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String subnodestring = subnodecode.replaceAll("\\s+", "");
		return nodestring.contains(subnodestring);
	}
	
	public static boolean containedByLines(Node node, int beginline, int endline, String code) {
		if (! node.getRange().isPresent())
			return false;
		Range noderange = node.getRange().get();
		if (noderange.begin.line < beginline || noderange.end.line > endline)
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String string = code.replaceAll("\\s+", "");
		return string.contains(nodestring);
	}
	
	private static void findSubNode_(Node node, int beginline, int endline, String subnodecode, Node[] subnode) {
		if (containLines(node, beginline, endline, subnodecode)) {
			subnode[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findSubNode_(child, beginline, endline, subnodecode, subnode);
			}
		}
	}
	
	public static Node findSubNode(Node node, int beginline, int endline, String subnodecode) {
		Node[] subnode = new Node[1];
		findSubNode_(node, beginline, endline, subnodecode, subnode);
		return subnode[0];
	}
	
	public static void removeComments(Node node) {
		node = node.removeComment();
		for (Comment comment : node.getAllContainedComments()) {
            comment.remove();
        }
		for (Node child : node.getChildNodes()) {
			removeComments(child);
		}
	}
	
	public static void removeAnnotation(Node node) {
		for (Node annotation : node.findAll(AnnotationExpr.class))
			annotation.removeForced();
	}
	
	public static MyNode calculateDepth(Node node, int depth) {
		MyNode mynode = new MyNode(node, depth);
		for (Node child : node.getChildNodes()) {
			MyNode mychild = calculateDepth(child, depth + 1);
			mynode.addChild(mychild);
		}
		return mynode;
	}
	
	public static ArrayList<String> analysisImports(MyNode myroot){
		ArrayList<String> imports = new ArrayList<String>();
		ArrayList<MyNode> dfs = myroot.DFS();
		for (MyNode mynode : dfs) {
			if (mynode.classEquals(ImportDeclaration.class)) {
				imports.add(mynode.getChildrenByTpye(Name.class).get(0).getValue());
			}
		}
		return imports;
	}
	
	public static String analysisPackages(MyNode myroot) {
		ArrayList<MyNode> dfs = myroot.DFS();
		for (MyNode mynode : dfs) {
			if (mynode.classEquals(PackageDeclaration.class)) {
				 return mynode.getChildrenByTpye(Name.class).get(0).getValue();
			}
		}
		return null;
	}
	
    public static void findCoveringBuggyNode_(Node node, String code, int beginline, int endline, Node[] subnode) {
		if (containLines(node, beginline, endline, code)) {
			subnode[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findCoveringBuggyNode_(child, code, beginline, endline, subnode);
			}
		}
	}
	
	public static Node findCoveringBuggyNode(Node context, String code, int beginline, int endline) throws IOException {
		Node[] subnode = {null};
		findCoveringBuggyNode_(context, code, beginline, endline, subnode);
		return subnode[0];
	}
	
	public static void findCoveringBuggyNode_(Node node, ArrayList<Node> subnodes, Node[] result) {
		boolean covering = true;
		for (Node subnode : subnodes) {
			Range range = subnode.getRange().get();
			if (! node.getRange().get().contains(range)) {
				covering = false;
				break;
			}
		}
		if (covering) {
			result[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findCoveringBuggyNode_(child, subnodes, result);
			}
		}
	}
	
	public static Node findCoveringBuggyNode(Node context, ArrayList<Node> subnodes) {
		Node[] result = {null};
		findCoveringBuggyNode_(context, subnodes, result);
		return result[0];
	}
	
	public static void findCoveredBuggyNode_(Node node, String code, int beginline, int endline, ArrayList<Node> subnodes) {
		if (containedByLines(node, beginline, endline, code)) {
			subnodes.add(node);
		}
		else {
			for (Node child : node.getChildNodes()) {
				findCoveredBuggyNode_(child, code, beginline, endline, subnodes);
			}
		}
	}
	
	public static void findCoveredBuggyNode_(Node node, int startLine, int startColumn, int endLine, int endColumn, ArrayList<Node> subnodes) {
		boolean covered = false;
		if ( !node.getRange().isPresent())
			return;
		Range range = node.getRange().get();
		if (range.begin.line > startLine || (range.begin.line == startLine && range.begin.column >= startColumn)) {
			if (range.end.line < endLine || (range.end.line == endLine && range.end.column <= endColumn))
				covered = true;
		}
		if (covered)
			subnodes.add(node);
		else {
			for (Node child : node.getChildNodes()) {
				findCoveredBuggyNode_(child, startLine, startColumn, endLine, endColumn, subnodes);
			}
		}
	}
	
	public static ArrayList<Node> findCoveredBuggyNode(Node context, int startLine, int startColumn, int endLine, int endColumn) {
		ArrayList<Node> subnodes = new ArrayList<Node>();
		findCoveredBuggyNode_(context, startLine, startColumn, endLine, endColumn, subnodes);
		return subnodes;
	}
	
	public static ArrayList<Node> findCoveredBuggyNode(Node context, int beginline, int endline, String code) {
		ArrayList<Node> subnodes = new ArrayList<Node>();
		findCoveredBuggyNode_(context, code, beginline, endline, subnodes);
		return subnodes;
	}
	
	public static HashMap<Node, Integer> findBuggyNodeIndex(Node context, ArrayList<Node> buggynodes){
		HashMap<Node, Integer> indices = new HashMap<Node, Integer>();
		for (Node buggynode : buggynodes) {
			String buggycode = buggynode.toString().trim();
			HashMap<String, Node> range2nodes = new HashMap<String, Node>();
			MyNode mycontext = calculateDepth(context, 0);
			for (MyNode mynode : mycontext.DFS()) {
				String nodecode = mynode.node.toString().trim();
				if (nodecode.equals(buggycode))
					range2nodes.put(mynode.node.getRange().get().toString(), mynode.node);
			}
			ArrayList<Node> samenodes = new ArrayList<Node>();
			for (Node node : range2nodes.values())
				samenodes.add(node);
			Collections.sort(samenodes, new Comparator<Node>() {
				 
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

			for (int i = 0; i < samenodes.size(); i += 1) {
				if (samenodes.get(i).toString().equals(buggynode.toString()) && 
						samenodes.get(i).getRange().get().equals(buggynode.getRange().get())) {
					indices.put(buggynode, i);
					break;
				}
			}
		}
		return indices;
	}
	
	public static ArrayList<String> findContextBeforeAfter(String filePath, Node context, int startLine, int startColumn, int endLine, int endColumn) throws IOException{
		ArrayList<String> contextBeforeAfter = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
		String line;
		int cnt = 0;
		String contextBefore = "", contextAfter = "";
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (cnt < context.getRange().get().begin.line || cnt > context.getRange().get().end.line)
				continue;
			if (cnt < startLine) {
				contextBefore += line;
				continue;
			}
			if (cnt == startLine) {
				for (int i = 0; i < line.length(); i += 1) {
					if (i + 1 < startColumn)
						contextBefore += line.substring(i, i + 1);
				}
			}
			if (cnt == endLine) {
				for (int i = 0; i < line.length(); i += 1) {
					if (i + 1 > endColumn)
						contextAfter += line.substring(i, i + 1);
				}
			}
			if (cnt > endLine) {
				contextAfter += line;
			}
		}
		br.close();
		contextBeforeAfter.add(contextBefore);
		contextBeforeAfter.add(contextAfter);
		return contextBeforeAfter;
	}
	
	public static ArrayList<Integer> findContextRange(String filePath, Node context, String contextBefore, String contextAfter) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
		String line;
		String content = "";
		int cnt = 0;
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (cnt < context.getRange().get().begin.line || cnt > context.getRange().get().end.line)
				continue;
			content += line;
		}
		br.close();
		
		ArrayList<Integer> range = new ArrayList<Integer>();
		br = new BufferedReader(new FileReader(new File(filePath)));
		cnt = 0;
		String before = "", after = content;
		int beforeLine = -1, beforeColumn = -1, afterLine = -1, afterColumn = -1;
		contextBefore = contextBefore.replaceAll("\\s+", "");
		contextAfter = contextAfter.replaceAll("\\s+", "");
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (cnt < context.getRange().get().begin.line || cnt > context.getRange().get().end.line)
				continue;
			if (beforeLine == -1) {
				for (int i = 0; i < line.length(); i += 1) {
					before += line.substring(i, i + 1);
					if (before.replaceAll("\\s+", "").equals(contextBefore)) {
						beforeLine = cnt;
						beforeColumn = i + 1;
						break;
					}
				}
			}
			if (afterLine == -1) {
				for (int i = 0; i < line.length(); i += 1) {
					if (after.replaceAll("\\s+", "").equals(contextAfter)) {
						afterLine = cnt;
						afterColumn = i + 1;
						break;
					}
					after = after.substring(1);
				}
			}
		}
		br.close();
		range.add(beforeLine);
		range.add(beforeColumn);
		range.add(afterLine);
		range.add(afterColumn);
		return range;
	}
	
	public static void findContext_(Node node, int beginline, int endline, String code, Node[] subnode) {
		if (containLines(node, beginline, endline, code)) {
			if (node.getClass().equals(MethodDeclaration.class) || node.getClass().equals(ConstructorDeclaration.class))
				subnode[1] = node;
			subnode[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findContext_(child, beginline, endline, code, subnode);
			}
		}
	}
	
	public static Node findContextNode(Node node, int beginline, int endline, String code) {
		Node[] subnode = {null, null};
		findContext_(node, beginline, endline, code, subnode);
		return subnode[1];
	}
	
	public static Node findContextNode(String filepath, int beginline, int endline) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		String line;
		int cnt = 0;
		String code = "";
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (beginline <= cnt && cnt < endline)
				code += line.trim() + " ";
			if (cnt >= endline)
				break;
		}
		br.close();
		code = code.trim();
		try {
			Node root = MyParser.parseFile(filepath);
			return findContextNode(root, beginline, endline, code);
		} catch (TokenMgrError e) {
			System.out.println(e.getStackTrace());
			return null;
		}
	}
	
	public static String findContext(String filepath, int beginline, int endline) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		String line;
		int cnt = 0;
		String code = "";
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (beginline <= cnt && cnt < endline)
				code += line.trim() + " ";
			if (cnt >= endline)
				break;
		}
		br.close();
		code = code.trim();
		
		try {
			Node root = MyParser.parseFile(filepath);
			Node context =  findContextNode(root, beginline, endline - 1, code);
			return context == null ? "" : context.toString();
		} catch (TokenMgrError e) {
			return "";
		}
	}

	public static void main(String[] args) throws Exception {
		Node root = MyParser.parseFile("D:\\java-eclipse-workspace\\deeprepair-javaparser\\src\\test\\java\\jiang719\\test\\Test1.java");
		// System.out.println(root);
		for (MyNode node : MyParser.calculateDepth(root, 0).DFS()) {
			System.out.println(node);
		}
		
	}
}
