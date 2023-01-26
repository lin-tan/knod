package jiang719;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Queue;
import java.util.Stack;
import java.util.concurrent.LinkedBlockingQueue;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

public class MyNode implements Comparable<MyNode>{
	public Node node;
	public final int depth;
	public MyNode father;
	public ArrayList<MyNode> children = new ArrayList<MyNode>();
	
	public MyNode(Node node, int depth){
		this.node = node;
		this.depth = depth;
	}
	
	public void addChild(MyNode child) {
		this.children.add(child);
		child.addFather(this);
	}
	
	public void addFather(MyNode father) {
		this.father = father;
	}
	
	public Range getRange() {
		return this.node.getRange().get();
	}
	
	public ArrayList<MyNode> getChildrenByTpye(Class<?> clazz) {
		ArrayList<MyNode> children = new ArrayList<MyNode>();
		for (MyNode child : this.children) {
			if (child.node.getClass().equals(clazz)) {
				children.add(child);
			}
		}
		return children;
	}
	
	public boolean classEquals(Class<?> clazz) {
		return this.node.getClass().equals(clazz);
	}
	
	public String getValue() {
		return this.node.toString();
	}
	
	public String getNodeClass() {
		String classname = this.node.getClass().toString();
		return classname.substring(classname.lastIndexOf(".") + 1);
	}
	
	public void sortChildren(MyNode node) {
		Collections.sort(node.children);
		for (MyNode child : node.children)
			sortChildren(child);
	}
	
	public ArrayList<MyNode> BFS() {
		sortChildren(this);
		
		ArrayList<MyNode> bfs = new ArrayList<MyNode>();
		Queue<MyNode> queue = new LinkedBlockingQueue<MyNode>();
		queue.add(this);
		while (! queue.isEmpty()) {
			MyNode cur = queue.poll();
			for (MyNode child : cur.children) {
				queue.add(child);
			}
			
			bfs.add(cur);
		}
		return bfs;
	}
	
	public ArrayList<MyNode> DFS(){
		sortChildren(this);
		
		ArrayList<MyNode> dfs = new ArrayList<MyNode>();
		Stack<MyNode> stack = new Stack<MyNode>();
		stack.push(this);
		while (! stack.isEmpty()) {
			MyNode cur = stack.pop();
			for (int i = cur.children.size() - 1; i >= 0; i -= 1) {
				stack.push(cur.children.get(i));
			}
			
			dfs.add(cur);
		}
		return dfs;
	}
	
	public void print() {
		for (MyNode mynode : this.DFS()) {
			System.out.println(mynode);
		}
	}
	
	public String toString() {
		return this.depth + " " + this.getNodeClass() + " " + this.children.size() + "\n" + 
				this.node.getRange().get().toString() + "\n" + this.node.toString();
	}

	public int compareTo(MyNode mynode) {
		if (! this.node.getRange().isPresent())
			return 1;
		if (! mynode.node.getRange().isPresent())
			return -1;
		Range r1 = this.node.getRange().get();
		Range r2 = mynode.node.getRange().get();
		if (r1.begin.isBefore(r2.begin))
			return -1;
		else if (r1.begin.equals(r2.begin) && r1.end.isBefore(r2.end))
			return -1;
		else
			return 1;
	}
}
