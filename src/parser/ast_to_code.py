import os
import sys

PARSER_DIR = os.path.abspath(__file__)
PARSER_DIR = PARSER_DIR[: PARSER_DIR.rfind('/') + 1]
sys.path.append(PARSER_DIR + '../../')

import javalang.tree as tree

nonterminal_nodes = {
	'CompilationUnit': tree.CompilationUnit(),
	'Import': tree.Import(),
	'Documented': tree.Documented(),
	'Declaration': tree.Declaration(),
	'TypeDeclaration': tree.TypeDeclaration(),
	'PackageDeclaration': tree.PackageDeclaration(),
	'ClassDeclaration': tree.ClassDeclaration(),
	'EnumDeclaration': tree.EnumDeclaration(),
	'InterfaceDeclaration': tree.InterfaceDeclaration(),
	'AnnotationDeclaration': tree.AnnotationDeclaration(),
	'Type': tree.Type(),
	'BasicType': tree.BasicType(),
	'ReferenceType': tree.ReferenceType(),
	'TypeArgument': tree.TypeArgument(),
	'TypeParameter': tree.TypeParameter(),
	'Annotation': tree.Annotation(),
	'ElementValuePair': tree.ElementValuePair(),
	'ElementArrayValue': tree.ElementArrayValue(),
	'Member': tree.Member(),
	'MethodDeclaration': tree.MethodDeclaration(),
	'FieldDeclaration': tree.FieldDeclaration(),
	'ConstructorDeclaration': tree.ConstructorDeclaration(),
	'ConstantDeclaration': tree.ConstantDeclaration(),
	'ArrayInitializer': tree.ArrayInitializer(),
	'VariableDeclaration': tree.VariableDeclaration(),
	'LocalVariableDeclaration': tree.LocalVariableDeclaration(),
	'VariableDeclarator': tree.VariableDeclarator(),
	'FormalParameter': tree.FormalParameter(),
	'InferredFormalParameter': tree.InferredFormalParameter(),
	'Statement': tree.Statement(),
	'IfStatement': tree.IfStatement(),
	'WhileStatement': tree.WhileStatement(),
	'DoStatement': tree.DoStatement(),
	'ForStatement': tree.ForStatement(),
	'AssertStatement': tree.AssertStatement(),
	'BreakStatement': tree.BreakStatement(),
	'ContinueStatement': tree.ContinueStatement(),
	'ReturnStatement': tree.ReturnStatement(),
	'ThrowStatement': tree.ThrowStatement(),
	'SynchronizedStatement': tree.SynchronizedStatement(),
	'TryStatement': tree.TryStatement(),
	'SwitchStatement': tree.SwitchStatement(),
	'BlockStatement': tree.BlockStatement(),
	'StatementExpression': tree.StatementExpression(),
	'TryResource': tree.TryResource(),
	'CatchClause': tree.CatchClause(),
	'CatchClauseParameter': tree.CatchClauseParameter(),
	'SwitchStatementCase': tree.SwitchStatementCase(),
	'ForControl': tree.ForControl(),
	'EnhancedForControl': tree.EnhancedForControl(),
	'Expression': tree.Expression(),
	'Assignment': tree.Assignment(),
	'TernaryExpression': tree.TernaryExpression(),
	'BinaryOperation': tree.BinaryOperation(),
	'Cast': tree.Cast(),
	'MethodReference': tree.MethodReference(),
	'LambdaExpression': tree.LambdaExpression(),
	'Primary': tree.Primary(),
	'Literal': tree.Literal(),
	'This': tree.This(),
	'MemberReference': tree.MemberReference(),
	'Invocation': tree.Invocation(),
	'ExplicitConstructorInvocation': tree.ExplicitConstructorInvocation(),
	'SuperConstructorInvocation': tree.SuperConstructorInvocation(),
	'MethodInvocation': tree.MethodInvocation(),
	'SuperMethodInvocation': tree.SuperMethodInvocation(),
	'SuperMemberReference': tree.SuperMemberReference(),
	'ArraySelector': tree.ArraySelector(),
	'ClassReference': tree.ClassReference(),
	'VoidClassReference': tree.VoidClassReference(),
	'Creator': tree.Creator(),
	'ArrayCreator': tree.ArrayCreator(),
	'ClassCreator': tree.ClassCreator(),
	'InnerClassCreator': tree.InnerClassCreator(),
	'EnumBody': tree.EnumBody(),
	'EnumConstantDeclaration': tree.EnumConstantDeclaration(),
	'AnnotationMethod': tree.AnnotationMethod(),
}
