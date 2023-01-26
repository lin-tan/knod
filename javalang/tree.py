
from .ast import Node

# ------------------------------------------------------------------------------


class WrappedNode:
    def __init__(self, node, attr):
        self.node = node
        self.attr = attr

    def to_code(self, show_plh=False):
        if not self.node:
            if show_plh and self.attr:
                return "<PHL_" + self.attr + "> "
            return " "
        elif isinstance(self.node, str):
            return self.node
        else:
            assert isinstance(self.node, Node), str(type(self.node)) + "is not Node"
            return self.node.to_code(show_plh)


def check_semicolon(src):
    src = src.strip()
    if len(src) > 0 and src[-1] != ";":
        src += ";"
    return src


def to_list(field, required=False, attr=None):
    if not field and not required:
        return None
    if required:
        assert attr is not None
    field_list = field
    if type(field) == set:
        field_list = list(field)
    elif type(field) != list:
        field_list = [field]
    return [WrappedNode(field, attr) for field in field_list]


class CompilationUnit(Node):
    attrs = ("package", "imports", "types")

    def to_code(self, show_plh=False):
        """
        package ...;
        imports ...;
        types ...;
        """
        package = WrappedNode(getattr(self, "package"), "package")      # optional
        imports = getattr(self, "imports")      # optional
        imports = to_list(imports)
        types = getattr(self, "types")          # optional
        types = to_list(types)

        result = ''
        if package.node:
            result += package.to_code(show_plh=show_plh)
        if imports:
            result += ''.join([im.to_code(show_plh=show_plh) for im in imports])
        if types:
            result += ''.join([ty.to_code(show_plh=show_plh) for ty in types])
        return result


class Import(Node):
    attrs = ("path", "static", "wildcard")

    def to_code(self, show_plh=False):
        """
        import static? path.*?;
        """
        path = WrappedNode(getattr(self, "path"), "path")       # required
        static = "static " if getattr(self, "static") else ""
        wildcard = ".*" if getattr(self, "wildcard") else ""
        return "import " + static + path.to_code(show_plh=show_plh) + wildcard + ";\n"


class Documented(Node):
    attrs = ("documentation",)


class Declaration(Node):
    attrs = ("modifiers", "annotations")


class TypeDeclaration(Declaration, Documented):
    attrs = ("name", "body")

    @property
    def fields(self):
        return [decl for decl in self.body if isinstance(decl, FieldDeclaration)]

    @property
    def methods(self):
        return [decl for decl in self.body if isinstance(decl, MethodDeclaration)]

    @property
    def constructors(self):
        return [decl for decl in self.body if isinstance(decl, ConstructorDeclaration)]


class PackageDeclaration(Declaration, Documented):
    attrs = ("name",)

    def to_code(self, show_plh=False):
        """
        package name;
        """
        name = WrappedNode(getattr(self, "name"), "name")   # required
        return "package " + name.to_code(show_plh=show_plh) + ";\n"


class ClassDeclaration(TypeDeclaration):
    attrs = ("type_parameters", "extends", "implements")

    def to_code(self, show_plh=False):
        """
        annotations modifiers class Name<A, B> extends C implements D, E {
            body
        }
        """
        annotations = getattr(self, "annotations")                      # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")                          # optional
        modifiers = to_list(modifiers)
        name = WrappedNode(getattr(self, "name"), "name")    # required
        type_parameters = getattr(self, "type_parameters")              # optional
        type_parameters = to_list(type_parameters)
        extends = WrappedNode(getattr(self, "extends"), "extends")      # optional
        implements = getattr(self, "implements")                        # optional
        implements = to_list(implements)
        bodys = getattr(self, "body")                                   # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " class " + name.to_code(show_plh=show_plh)
        if type_parameters:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_parameters]) + ">"
        if extends.node:
            result += " extends " + extends.to_code(show_plh=show_plh)
        if implements:
            result += " implements " + ",".join([im.to_code(show_plh=show_plh) for im in implements])
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result


class EnumDeclaration(TypeDeclaration):
    attrs = ("implements",)

    @property
    def fields(self):
        return [decl for decl in self.body.declarations if isinstance(decl, FieldDeclaration)]

    @property
    def methods(self):
        return [decl for decl in self.body.declarations if isinstance(decl, MethodDeclaration)]

    def to_code(self, show_plh=False):
        """
        annotations modifiers enum Name implements A, B {
            body
        }
        """
        annotations = getattr(self, "annotations")          # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")              # optional
        modifiers = to_list(modifiers)
        name = WrappedNode(getattr(self, "name"))    # required
        implements = getattr(self, "implements")            # optional
        implements = to_list(implements)
        bodys = getattr(self, "body")                       # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " enum " + name.to_code(show_plh=show_plh)
        if implements:
            result += " implements " + ",".join([im.to_code(show_plh=show_plh) for im in implements])
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result


class InterfaceDeclaration(TypeDeclaration):
    attrs = ("type_parameters", "extends",)

    def to_code(self, show_plh=False):
        """
        annotations modifiers interface Name<A, B> extends C, D {
            body
        }
        """
        annotations = getattr(self, "annotations")      # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")          # optional
        modifiers = to_list(modifiers)
        name = WrappedNode(getattr(self, "name"), "name")    # required
        type_parameters = getattr(self, "type_parameters")  # optional
        type_parameters = to_list(type_parameters)
        extends = WrappedNode(getattr(self, "extends"), "extends")  # optional
        bodys = getattr(self, "body")                   # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " interface " + name.to_code(show_plh=show_plh)
        if type_parameters:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_parameters]) + ">"
        if extends.node:
            result += " extends " + extends.to_code(show_plh=show_plh)
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result


class AnnotationDeclaration(TypeDeclaration):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        annotations @interface name { body }
        """
        annotations = getattr(self, "annotations")      # optional
        annotations = to_list(annotations)
        name = WrappedNode(getattr(self, "name"), "name")    # required
        bodys = getattr(self, "body")                   # optional
        bodys = to_list(bodys)
        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " @interface " + name.to_code(show_plh=show_plh)
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result

# ------------------------------------------------------------------------------


class Type(Node):
    attrs = ("name", "dimensions",)


class BasicType(Type):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        name[dimension]...
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        dimensions = getattr(self, "dimensions")    # optional
        dimensions = to_list(dimensions)
        result = name.to_code(show_plh=show_plh)
        if dimensions:
            for dim in dimensions:
                result += "[]" if (dim.node is None or dim.to_code(show_plh=show_plh) == "None") \
                    else "[" + dim.to_code(show_plh=show_plh) + "]"
        return result


class ReferenceType(Type):
    attrs = ("arguments", "sub_type")

    def to_code(self, show_plh=False):
        """
        name.sub_type<arguments>[dimension]...
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        dimensions = getattr(self, "dimensions")    # optional
        dimensions = to_list(dimensions)
        sub_types = getattr(self, "sub_type")       # optional
        sub_types = to_list(sub_types)
        arguments = getattr(self, "arguments")      # optional
        arguments = to_list(arguments)

        result = name.to_code(show_plh=show_plh)
        if sub_types:
            for sub_type in sub_types:
                result += "." + sub_type.to_code(show_plh=show_plh)
        if arguments:
            result += "<" + ",".join([arg.to_code(show_plh=show_plh) for arg in arguments]) + ">"
        if dimensions:
            for dim in dimensions:
                result += "[]" if (dim.node is None or dim.to_code(show_plh=show_plh) == "None") \
                    else "[" + dim.to_code(show_plh=show_plh) + "]"
        return result


class TypeArgument(Node):
    attrs = ("type", "pattern_type")

    def to_code(self, show_plh=False):
        """
        T, ?, ? super|extends T
        """
        pattern_type = getattr(self, "pattern_type")        # optional
        ty = WrappedNode(getattr(self, "type"), "type")     # optional
        ty = ty.to_code(show_plh=show_plh) if ty.node else ""
        if pattern_type == "?":
            return "?"
        elif pattern_type in ["super", "extends"]:
            return "? " + pattern_type + " " + ty
        else:
            return ty

# ------------------------------------------------------------------------------


class TypeParameter(Node):
    attrs = ("name", "extends")

    def to_code(self, show_plh=False):
        """
        name extends A & B ...
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        extends = getattr(self, "extends")  # optional
        extends = to_list(extends)
        result = name.to_code(show_plh=show_plh)
        if extends:
            result += " extends " + " & ".join([ex.to_code(show_plh=show_plh) for ex in extends])
        return result


# ------------------------------------------------------------------------------

class Annotation(Node):
    attrs = ("name", "element")

    def to_code(self, show_plh=False):
        """
        @name, @name(E), @name({E,...}), @name(A=B, C=D,...)
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        result = "@" + name.to_code(show_plh=show_plh)
        elements = getattr(self, "element")     # optional
        elements = to_list(elements)
        if elements:
            result += "(" + ",".join([e.to_code(show_plh=show_plh) for e in elements]) + ")"
        return result


class ElementValuePair(Node):
    attrs = ("name", "value")

    def to_code(self, show_plh=False):
        """
        name=value
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        value = WrappedNode(getattr(self, "value"), "value")     # required
        return name.to_code(show_plh=show_plh) + "=" + value.to_code(show_plh=show_plh)


class ElementArrayValue(Node):
    attrs = ("values",)

    def to_code(self, show_plh=False):
        """
        {A, B,...}
        """
        values = getattr(self, "values")    # optional
        values = to_list(values)
        if values:
            return "{" + ",".join([value.to_code(show_plh=show_plh) for value in values]) + "}"
        return "{}"


# ------------------------------------------------------------------------------

class Member(Documented):
    attrs = ()


class MethodDeclaration(Member, Declaration):
    attrs = ("type_parameters", "return_type", "name", "parameters", "throws", "body")

    def to_code(self, show_plh=False):
        """
        annotations modifiers type_parameters return_type name(parameters) throws {
            body
        }
        """
        annotations = getattr(self, "annotations")      # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")          # optional
        modifiers = to_list(modifiers)
        type_parameters = getattr(self, "type_parameters")  # optional
        type_parameters = to_list(type_parameters)
        return_type = WrappedNode(getattr(self, "return_type"), "return_type")   # optional
        name = WrappedNode(getattr(self, "name"), "name")    # required
        parameters = getattr(self, "parameters")        # optional
        parameters = to_list(parameters)
        throws = getattr(self, "throws")                # optional
        throws = to_list(throws)
        bodys = getattr(self, "body")                   # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        if type_parameters:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_parameters]) + ">"
        result += " "
        if return_type.node:
            result += return_type.to_code(show_plh=show_plh)
        else:
            result += "void"
        result += " "
        result += name.to_code(show_plh=show_plh)
        if parameters:
            result += "(" + ",".join([param.to_code(show_plh=show_plh) for param in parameters]) + ")"
        else:
            result += "()"
        if throws:
            result += " throws " + ",".join([throw.to_code(show_plh=show_plh) for throw in throws])
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result


class FieldDeclaration(Member, Declaration):
    attrs = ("type", "declarators")

    def to_code(self, show_plh=False):
        """
        annotations modifiers type declarators...
        """
        annotations = getattr(self, "annotations")      # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")  # required
        declarators = getattr(self, "declarators")
        declarators = to_list(declarators, required=True, attr="declarators")   # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " "
        result += ",".join([declarator.to_code(show_plh=show_plh) for declarator in declarators])
        result += ";\n"
        return result


class ConstructorDeclaration(Declaration, Documented):
    attrs = ("type_parameters", "name", "parameters", "throws", "body")

    def to_code(self, show_plh=False):
        """
        annotations modifiers type_parameters name(parameters) throws {
            body
        }
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        type_parameters = getattr(self, "type_parameters")  # optional
        type_parameters = to_list(type_parameters)
        name = WrappedNode(getattr(self, "name"), "name")  # required
        parameters = getattr(self, "parameters")  # optional
        parameters = to_list(parameters)
        throws = getattr(self, "throws")  # optional
        throws = to_list(throws)
        bodys = getattr(self, "body")  # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        if type_parameters:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_parameters]) + ">"
        result += " " + name.to_code(show_plh=show_plh)
        if parameters:
            result += "(" + ",".join([param.to_code(show_plh=show_plh) for param in parameters]) + ")"
        else:
            result += "()"
        if throws:
            result += " throws " + ",".join([throw.to_code(show_plh=show_plh) for throw in throws])
        result += "{\n"
        if bodys:
            result += "".join([body.to_code(show_plh=show_plh) for body in bodys])
        result += "}\n"
        return result


# ------------------------------------------------------------------------------

class ConstantDeclaration(FieldDeclaration):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        annotations modifiers type declarators...
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")  # required
        declarators = getattr(self, "declarators")
        declarators = to_list(declarators, required=True, attr="declarators")  # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " "
        result += ",".join([declarator.to_code(show_plh=show_plh) for declarator in declarators])
        result += ";\n"
        return result


class ArrayInitializer(Node):
    attrs = ("initializers",)

    def to_code(self, show_plh=False):
        """
        {1, 2, 3...}
        """
        initializers = getattr(self, "initializers")    # optional
        initializers = to_list(initializers)
        if initializers:
            return "{" + ",".join([init.to_code(show_plh=show_plh) for init in initializers]) + "}"
        else:
            return "{}"


class VariableDeclaration(Declaration):
    attrs = ("type", "declarators")

    def to_code(self, show_plh=False):
        """
        annotations modifiers type declarators...
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")  # required
        declarators = getattr(self, "declarators")
        declarators = to_list(declarators, required=True, attr="declarators")  # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " "
        result += ",".join([declarator.to_code(show_plh=show_plh) for declarator in declarators])
        result += ";\n"
        return result


class LocalVariableDeclaration(VariableDeclaration):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        annotations modifiers type declarators...
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")  # required
        declarators = getattr(self, "declarators")
        declarators = to_list(declarators, required=True, attr="declarators")  # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " "
        result += ",".join([declarator.to_code(show_plh=show_plh) for declarator in declarators])
        result += ";\n"
        return result


class VariableDeclarator(Node):
    attrs = ("name", "dimensions", "initializer")

    def to_code(self, show_plh=False):
        """
        name[dimensions] = initializer
        """
        name = WrappedNode(getattr(self, "name"), "name")    # required
        dimensions = getattr(self, "dimensions")    # optional
        dimensions = to_list(dimensions)
        initializer = getattr(self, "initializer")  # optional

        result = name.to_code(show_plh=show_plh)
        if dimensions:
            for dim in dimensions:
                result += "[]" if (dim.node is None or dim.to_code(show_plh=show_plh) == "None") \
                    else "[" + dim.to_code(show_plh=show_plh) + "]"
        if initializer:
            result += " = " + initializer.to_code(show_plh=show_plh)
        return result


class FormalParameter(Declaration):
    attrs = ("type", "name", "varargs")

    def to_code(self, show_plh=False):
        """
        type name, type... name
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")  # required
        name = WrappedNode(getattr(self, "name"), "name")    # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " "
        if getattr(self, "varargs"):
            return result + "... " + name.to_code(show_plh=show_plh)
        else:
            return result + " " + name.to_code(show_plh=show_plh)


class InferredFormalParameter(Node):
    attrs = ('name',)

    def to_code(self, show_plh=False):
        name = WrappedNode(getattr(self, "name"), "name")  # required
        return name.to_code(show_plh=show_plh)


# ------------------------------------------------------------------------------

class Statement(Node):
    attrs = ("label",)

    def to_code(self, show_plh=False):
        return ""


class IfStatement(Statement):
    attrs = ("condition", "then_statement", "else_statement")

    def to_code(self, show_plh=False):
        """
        if (condition) then_statement else else_statement
        """
        label = WrappedNode(getattr(self, "label"), "label")    # optional
        condition = WrappedNode(getattr(self, "condition"), "condition")    # required
        then_statement = WrappedNode(getattr(self, "then_statement"), "then_statement")     # required
        else_statement = WrappedNode(getattr(self, "else_statement"), "else_statement")     # optional
        result = "if (" + condition.to_code(show_plh=show_plh) + " )" + then_statement.to_code(show_plh=show_plh)
        if else_statement.node:
            result += "else " + else_statement.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class WhileStatement(Statement):
    attrs = ("condition", "body")

    def to_code(self, show_plh=False):
        """
        label : while ( condition ) body
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        condition = WrappedNode(getattr(self, "condition"), "condition")  # required
        body = WrappedNode(getattr(self, "body"), "body")       # required

        result = "while (" + condition.to_code(show_plh=show_plh) + ")\n" + body.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class DoStatement(Statement):
    attrs = ("condition", "body")

    def to_code(self, show_plh=False):
        """
        label : do body while (condition);
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        condition = WrappedNode(getattr(self, "condition"), "condition")  # required
        body = WrappedNode(getattr(self, "body"), "body")  # required

        result = "do\n" + body.to_code(show_plh=show_plh) + "while ( " + condition.to_code(show_plh=show_plh) + ");\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class ForStatement(Statement):
    attrs = ("control", "body")

    def to_code(self, show_plh=False):
        """
        label : for (control) body
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        control = WrappedNode(getattr(self, "control"), "control")  # required
        body = WrappedNode(getattr(self, "body"), "body")  # required

        result = "for (" + control.to_code(show_plh=show_plh) + ")\n" + body.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class AssertStatement(Statement):
    attrs = ("condition", "value")

    def to_code(self, show_plh=False):
        """
        assert condition : value;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        condition = WrappedNode(getattr(self, "condition"), "condition")  # required
        value = WrappedNode(getattr(self, "value"), "value")    # optional

        result = "assert " + condition.to_code(show_plh=show_plh)
        if value.node:
            result += " : " + value.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        result = check_semicolon(result) + "\n"
        return result


class BreakStatement(Statement):
    attrs = ("goto",)

    def to_code(self, show_plh=False):
        """
        break goto;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        goto = WrappedNode(getattr(self, "goto"), "goto")       # optional
        if goto.node:
            result = "break " + goto.to_code(show_plh=show_plh) + ";\n"
        else:
            result = "break;\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class ContinueStatement(Statement):
    attrs = ("goto",)

    def to_code(self, show_plh=False):
        """
        continue goto;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        goto = WrappedNode(getattr(self, "goto"), "goto")  # optional
        if goto:
            result = "continue " + goto.to_code(show_plh=show_plh) + ";\n"
        else:
            result = "continue;\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class ReturnStatement(Statement):
    attrs = ("expression",)

    def to_code(self, show_plh=False):
        """
        return expression;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        expr = WrappedNode(getattr(self, "expression"), "expression")     # optional
        if expr.node:
            result = "return " + expr.to_code(show_plh=show_plh)
        else:
            result = "return"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        result = check_semicolon(result) + "\n"
        return result


class ThrowStatement(Statement):
    attrs = ("expression",)

    def to_code(self, show_plh=False):
        """
        throw value;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        expr = WrappedNode(getattr(self, "expression"), "expression")  # required
        result = "throw " + expr.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        result = check_semicolon(result) + "\n"
        return result


class SynchronizedStatement(Statement):
    attrs = ("lock", "block")

    def to_code(self, show_plh=False):
        """
        synchronized (lock) {
            block
        }
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        lock = WrappedNode(getattr(self, "lock"), "lock")   # required
        block = getattr(self, "block")      # optional
        block = to_list(block)
        result = "synchronized(" + lock.to_code(show_plh=show_plh) + "){\n"
        if block:
            result += "".join([statement.to_code(show_plh=show_plh) for statement in block])
        result += "}\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class TryStatement(Statement):
    attrs = ("resources", "block", "catches", "finally_block")

    def to_code(self, show_plh=False):
        """
        try {block} catches finally {finally_block}
        try (R r; S s) {block} ...
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        resources = getattr(self, "resources")      # optional
        resources = to_list(resources)
        block = getattr(self, "block")      # optional
        block = to_list(block)
        catches = getattr(self, "catches")  # optional
        catches = to_list(catches)
        finally_block = getattr(self, "finally_block")  # optional
        finally_block = to_list(finally_block)

        result = "try"
        if resources:
            result += "(" + ";".join(resource.to_code(show_plh=show_plh) for resource in resources) + ")"
        result += "{\n"
        if block:
            result += "".join([statement.to_code(show_plh=show_plh) for statement in block])
        result += "}\n"
        if catches:
            result += "".join([catch.to_code(show_plh=show_plh) for catch in catches])
        if finally_block:
            result += "finally {\n" + "".join([statement.to_code(show_plh=show_plh) for statement in finally_block]) + "}\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class SwitchStatement(Statement):
    attrs = ("expression", "cases")

    def to_code(self, show_plh=False):
        """
        switch (expression) {
            cases
        }
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        expr = WrappedNode(getattr(self, "expression"), "expression")   # required
        cases = getattr(self, "cases")  # optional
        cases = to_list(cases)
        result = "switch (" + expr.to_code(show_plh=show_plh) + ") {\n"
        if cases:
            result += "".join([case.to_code(show_plh=show_plh) for case in cases])
        result += "}\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class BlockStatement(Statement):
    attrs = ("statements",)

    def to_code(self, show_plh=False):
        """
        { statements }
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        statements = getattr(self, "statements")    # optional
        statements = to_list(statements)
        result = "{\n"
        if statements:
            result += "".join([statement.to_code(show_plh=show_plh) for statement in statements])
        result += "}\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class StatementExpression(Statement):
    attrs = ("expression",)

    def to_code(self, show_plh=False):
        """
        expression;
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        expr = WrappedNode(getattr(self, "expression"), "expression")   # requires
        result = expr.to_code(show_plh=show_plh)
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        result = check_semicolon(result) + "\n"
        return result


# ------------------------------------------------------------------------------

class TryResource(Declaration):
    attrs = ("type", "name", "value")

    def to_code(self, show_plh=False):
        """
        annotations modifiers type name = value
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        ty = WrappedNode(getattr(self, "type"), "type")     # required
        name = WrappedNode(getattr(self, "name"), "name")   # required
        value = WrappedNode(getattr(self, "value"), "value")    # optional

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += " " + ty.to_code(show_plh=show_plh) + " " + name.to_code(show_plh=show_plh)
        if value.node:
            result += " = " + value.to_code(show_plh=show_plh)
        return result


class CatchClause(Statement):
    attrs = ("parameter", "block")

    def to_code(self, show_plh=False):
        """
        catch (parameter) {
            block
        }
        """
        label = WrappedNode(getattr(self, "label"), "label")  # optional
        param = WrappedNode(getattr(self, "parameter"), "parameter")    # required
        block = getattr(self, "block")  # optional
        block = to_list(block)
        result = "catch (" + param.to_code(show_plh=show_plh) + ") {\n"
        if block:
            result += "".join([statement.to_code(show_plh=show_plh) for statement in block])
        result += "}\n"
        if label.node:
            result = label.to_code(show_plh=show_plh) + ": " + result
        return result


class CatchClauseParameter(Declaration):
    attrs = ("types", "name")

    def to_code(self, show_plh=False):
        """
        type1|type2 name
        """
        annotations = getattr(self, "annotations")  # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        types = getattr(self, "types")  # required
        types = to_list(types, required=True, attr="types")
        name = WrappedNode(getattr(self, "name"), "name")  # required

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += "|".join([ty.to_code(show_plh=show_plh) for ty in types])
        result += " " + name.to_code(show_plh=show_plh)
        return result


# ------------------------------------------------------------------------------

class SwitchStatementCase(Node):
    attrs = ("case", "statements")

    def to_code(self, show_plh=False):
        """
        case case1 : case case2 : { statements }
        """
        cases = getattr(self, "case")   # required
        cases = to_list(cases, required=True, attr="case")                      # list
        statements = getattr(self, "statements")    # optional
        statements = to_list(statements)

        result = ""
        for case in cases:
            result += "case " + case.to_code(show_plh=show_plh) + ": "
        if cases == []:
            result += "default: "
        if statements and len(statements) == 1 and isinstance(statements[0].node, BlockStatement):
            result += statements[0].to_code(show_plh=show_plh) + "\n"
        elif statements:
            # result += "{ " + "".join([statement.to_code(show_plh=show_plh) for statement in statements]) + "}\n"
            result += "".join([statement.to_code(show_plh=show_plh) for statement in statements]) + "\n"
        return result


class ForControl(Node):
    attrs = ("init", "condition", "update")

    def to_code(self, show_plh=False):
        """
        init; condition; update
        """
        init = getattr(self, "init")            # None, Node or list, optional
        init = to_list(init)
        condition = getattr(self, "condition")  # None or Node, optional
        condition = to_list(condition)
        update = getattr(self, "update")        # None, Node or list, optional
        update = to_list(update)

        result = ",".join([statement.to_code(show_plh=show_plh) for statement in init]) if init else ""
        result_init = result.strip() if len(result.strip()) > 0 and result.strip()[-1] == ";" \
            else result.strip() + ";"

        result = ",".join([statement.to_code(show_plh=show_plh) for statement in condition]) if condition else ""
        result_condition = result.strip() if len(result.strip()) > 0 and result.strip()[-1] == ";" \
            else result.strip() + ";"

        result = ",".join([statement.to_code(show_plh=show_plh) for statement in update]) if update else ""
        result_update = result

        return result_init + result_condition + result_update


class EnhancedForControl(Node):
    attrs = ("var", "iterable")

    def to_code(self, show_plh=False):
        """
        var : iterable
        """
        var = WrappedNode(getattr(self, "var"), "var").to_code(show_plh=show_plh)
        iterable = WrappedNode(getattr(self, "iterable"), "iterable").to_code(show_plh=show_plh)
        result = var.strip()[:-1] if len(var.strip()) > 0 and var.strip()[-1] == ";" else var.strip()
        return result + " : " + iterable


# ------------------------------------------------------------------------------

class Expression(Node):
    attrs = ()


class Assignment(Expression):
    attrs = ("expressionl", "value", "type")

    def to_code(self, show_plh=False):
        """
        expressionl type value
        """
        exprl = WrappedNode(getattr(self, "expressionl"), "expressionl").to_code(show_plh=show_plh)
        ty = WrappedNode(getattr(self, "type"), "type").to_code(show_plh=show_plh)
        value = WrappedNode(getattr(self, "value"), "value").to_code(show_plh=show_plh)
        return exprl + " " + ty + " " + value


class TernaryExpression(Expression):
    attrs = ("condition", "if_true", "if_false")

    def to_code(self, show_plh=False):
        """
        condition ? if_true : if_false
        """
        condition = WrappedNode(getattr(self, "condition"), "condition").to_code(show_plh=show_plh)
        if_true = WrappedNode(getattr(self, "if_true"), "if_true").to_code(show_plh=show_plh)
        if_false = WrappedNode(getattr(self, "if_false"), "if_false").to_code(show_plh=show_plh)
        return condition + " ? " + if_true + " : " + if_false


def is_prior(o1, o2):
    priority = {
        "*": 1, "/": 1, "%": 1,
        "+": 2, "-": 2,
        "<<": 3, ">>": 3, ">>>": 3,
        "<": 4, "<=": 4, ">": 4, ">=": 4, "instanceof": 4,
        "==": 5, "!=": 5,
        "&": 6, "^": 7, "|": 8, "&&": 9, "||": 10,
    }
    return priority[o1] < priority[o2]


class BinaryOperation(Expression):
    attrs = ("prefix_operators", "operator", "operandl", "operandr")

    def to_code(self, show_plh=False):
        """
        prefix_operators operandl operator operandr
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        operandl = WrappedNode(getattr(self, "operandl"), "oprandl")    # required
        operator = WrappedNode(getattr(self, "operator"), "operator")   # required
        operandr = WrappedNode(getattr(self, "operandr"), "operandr")   # required
        if (isinstance(operandl.node, BinaryOperation) and is_prior(operator.node, operandl.node.operator)) or \
                isinstance(operandl.node, (TernaryExpression, Assignment)):
            operandl = "(" + operandl.to_code(show_plh=show_plh) + ")"
        else:
            operandl = operandl.to_code(show_plh=show_plh)
        if (isinstance(operandr.node, BinaryOperation) and is_prior(operator.node, operandr.node.operator)) or \
                isinstance(operandr.node, (TernaryExpression, Assignment)):
            operandr = "(" + operandr.to_code(show_plh=show_plh) + ")"
        else:
            operandr = operandr.to_code(show_plh=show_plh)
        if prefix:
            result = "".join([pre.to_code(show_plh=show_plh) for pre in prefix]) + \
                     "(" + operandl + " " + operator.to_code(show_plh=show_plh) + " " + operandr + ")"
        else:
            result = operandl + " " + operator.to_code(show_plh=show_plh) + " " + operandr
        return result


class MethodReference(Expression):
    attrs = ("expression", "method", "type_arguments")

    def to_code(self, show_plh=False):
        """
        expression::<type_arguments> method
        """
        expression = WrappedNode(getattr(self, "expression"), "expression")     # required
        type_arguments = getattr(self, "type_arguments")    # optional
        type_arguments = to_list(type_arguments)
        method = WrappedNode(getattr(self, "method"), "method")     # required

        result = expression.to_code(show_plh=show_plh) + "::"
        if type_arguments:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_arguments]) + ">"
        result += method.to_code(show_plh=show_plh)
        return result


class LambdaExpression(Expression):
    attrs = ('parameters', 'body')

    def to_code(self, show_plh=False):
        """
        (parameters)->{body}
        (parameters)->body
        """
        parameters = getattr(self, "parameters")    # optional
        parameters = to_list(parameters)
        bodys = getattr(self, "body")   # required
        bodys = to_list(bodys, required=True, attr="body")

        result = "("
        if parameters:
            result += ",".join([param.to_code(show_plh=show_plh) for param in parameters])
        result += ")->"
        if bodys and len(bodys) > 1:
            result += "{" + "".join([statement.to_code(show_plh=show_plh) for statement in bodys]) + "}"
        else:
            result += bodys[0].to_code(show_plh=show_plh)
        return result


# ------------------------------------------------------------------------------

class Primary(Expression):
    attrs = ("prefix_operators", "postfix_operators", "qualifier", "selectors")


#class Cast(Expression):
class Cast(Primary):
    attrs = ("type", "expression")

    def to_code(self, show_plh=False):
        """
        (type) expression
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)

        ty = WrappedNode(getattr(self, "type"), "type")
        expression = WrappedNode(getattr(self, "expression"), "expression")

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."

        if isinstance(expression.node, BinaryOperation) or isinstance(expression.node, TernaryExpression):
            result += "(" + ty.to_code(show_plh=show_plh) + ") (" + expression.to_code(show_plh=show_plh) + ")"
        else:
            result += "(" + ty.to_code(show_plh=show_plh) + ")" + expression.to_code(show_plh=show_plh)

        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class Literal(Primary):
    attrs = ("value",)

    def to_code(self, show_plh=False):
        """
        value
        """
        prefix = getattr(self, "prefix_operators")      # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")    # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        value = WrappedNode(getattr(self, "value"), "value")    # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += value.to_code(show_plh=show_plh)
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class This(Primary):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        prefix_operators this.selector1.selector2 postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")    # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "this"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class MemberReference(Primary):
    attrs = ("member",)

    def to_code(self, show_plh=False):
        """
        prefix_operators qualifier.member selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        member = WrappedNode(getattr(self, "member"), "member")     # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += member.to_code(show_plh=show_plh)
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class Invocation(Primary):
    attrs = ("type_arguments", "arguments")


class ExplicitConstructorInvocation(Invocation):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        prefix_operators type_arguments this (arguments) selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)

        type_arguments = getattr(self, "type_arguments")    # optional
        type_arguments = to_list(type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if type_arguments:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_arguments]) + ">"
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "this"
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class SuperConstructorInvocation(Invocation):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        prefix_operators type_arguments super (arguments) selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)

        type_arguments = getattr(self, "type_arguments")  # optional
        type_arguments = to_list(type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if type_arguments:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_arguments]) + ">"
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "super"
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class MethodInvocation(Invocation):
    attrs = ("member",)

    def to_code(self, show_plh=False):
        """
        prefix_operators type_arguments qualifier.member (arguments) selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        type_arguments = getattr(self, "type_arguments")  # optional
        type_arguments = to_list(type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)

        member = WrappedNode(getattr(self, "member"), "member")     # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if type_arguments:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_arguments]) + ">"
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += member.to_code(show_plh=show_plh)
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class SuperMethodInvocation(Invocation):
    attrs = ("member",)

    def to_code(self, show_plh=False):
        """
        prefix_operators super.<type_arguments>member(arguments) selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        type_arguments = getattr(self, "type_arguments")  # optional
        type_arguments = to_list(type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)

        member = WrappedNode(getattr(self, "member"), "member")  # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "super."
        if type_arguments:
            result += "<" + ",".join([ty.to_code(show_plh=show_plh) for ty in type_arguments]) + ">"
        result += member.to_code(show_plh=show_plh)
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class SuperMemberReference(Primary):
    attrs = ("member",)

    def to_code(self, show_plh=False):
        """
        prefix_operators super.member selectors postfix_operators
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        member = WrappedNode(getattr(self, "member"), "member")   # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "super." + member.to_code(show_plh=show_plh)
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class ArraySelector(Expression):
    attrs = ("index",)

    def to_code(self, show_plh=False):
        """
        [index]
        """
        index = WrappedNode(getattr(self, "index"), "index")    # required
        return "[" + index.to_code(show_plh=show_plh) + "]"


class ClassReference(Primary):
    attrs = ("type",)

    def to_code(self, show_plh=False):
        """
        qualifier.type.class
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        ty = WrappedNode(getattr(self, "type"), "type")     # required

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += ty.to_code(show_plh=show_plh) + ".class"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class VoidClassReference(ClassReference):
    attrs = ()

    def to_code(self, show_plh=False):
        """
        void.class
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += "void.class"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


# ------------------------------------------------------------------------------

class Creator(Primary):
    attrs = ("type",)


class ArrayCreator(Creator):
    attrs = ("dimensions", "initializer")

    def to_code(self, show_plh=False):
        """
        new type[dimensions]
        new type[] {1,2,...}
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        ty = WrappedNode(getattr(self, "type"), "type")  # required

        dimensions = getattr(self, "dimensions")    # required
        dimensions = to_list(dimensions, required=True, attr="dimensions")
        initializer = WrappedNode(getattr(self, "initializer"), "initializer")  # optional

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        result += "new "
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        result += ty.to_code(show_plh=show_plh)
        for dim in dimensions:
            result += "[" + dim.to_code(show_plh=show_plh) + "]" if dim.node else "[]"
        if initializer.node:
            result += initializer.to_code(show_plh=show_plh)
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class ClassCreator(Creator):
    attrs = ("constructor_type_arguments", "arguments", "body")

    def to_code(self, show_plh=False):
        """
        new <constructor_type_arguments> type(arguments) {body} selectors
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        ty = WrappedNode(getattr(self, "type"), "type")  # required

        constructor_type_arguments = getattr(self, "constructor_type_arguments")    # optional
        constructor_type_arguments = to_list(constructor_type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)
        body = getattr(self, "body")    # optional
        body = to_list(body)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        result += "new "
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        if constructor_type_arguments:
            result += "<" + ",".join([argument.to_code(show_plh=show_plh)
                                      for argument in constructor_type_arguments]) + ">"
        result += ty.to_code(show_plh=show_plh)
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if body:
            result += "{\n" + "".join([statement.to_code(show_plh=show_plh) for statement in body]) + "}"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


class InnerClassCreator(Creator):
    attrs = ("constructor_type_arguments", "arguments", "body")

    def to_code(self, show_plh=False):
        """
        new <constructor_type_arguments> type(arguments) {body} selectors
        """
        prefix = getattr(self, "prefix_operators")  # optional
        prefix = to_list(prefix)
        postfix = getattr(self, "postfix_operators")  # optional
        postfix = to_list(postfix)
        qualifier = getattr(self, "qualifier")  # optional
        qualifier = to_list(qualifier)
        selectors = getattr(self, "selectors")  # optional
        selectors = to_list(selectors)
        ty = WrappedNode(getattr(self, "type"), "type")  # required

        constructor_type_arguments = getattr(self, "constructor_type_arguments")  # optional
        constructor_type_arguments = to_list(constructor_type_arguments)
        arguments = getattr(self, "arguments")  # optional
        arguments = to_list(arguments)
        body = getattr(self, "body")  # optional
        body = to_list(body)

        result = ""
        if prefix:
            result += "".join([pre.to_code(show_plh=show_plh) for pre in prefix])
        result += "new "
        if qualifier:
            for qu in qualifier:
                result += qu.to_code(show_plh=show_plh) + "."
        if constructor_type_arguments:
            result += "<" + ",".join([argument.to_code(show_plh=show_plh)
                                      for argument in constructor_type_arguments]) + ">"
        result += ty.to_code(show_plh=show_plh)
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if body:
            result += "{\n" + "".join([statement.to_code(show_plh=show_plh) for statement in body]) + "}"
        if selectors:
            for selector in selectors:
                if isinstance(selector.node, ArraySelector):
                    result += selector.to_code(show_plh=show_plh)
                else:
                    result += "." + selector.to_code(show_plh=show_plh)
        if postfix:
            result += "".join([post.to_code(show_plh=show_plh) for post in postfix])
        return result


# ------------------------------------------------------------------------------

class EnumBody(Node):
    attrs = ("constants", "declarations")

    def to_code(self, show_plh=False):
        """
        constant1, constant2...; declaration1; declaration2...
        """
        constants = getattr(self, "constants")          # optional
        constants = to_list(constants)
        declarations = getattr(self, "declarations")    # optional
        declarations = to_list(declarations)

        result = ""
        if constants:
            result += ",".join([constant.to_code(show_plh=show_plh) for constant in constants])
        if result.strip() != "" and declarations:
            result += ";"
        if declarations:
            result += "".join([declaration.to_code(show_plh=show_plh) for declaration in declarations])
        return result


class EnumConstantDeclaration(Declaration, Documented):
    attrs = ("name", "arguments", "body")

    def to_code(self, show_plh=False):
        """
        annotations modifiers name(arguments){body}
        """
        annotations = getattr(self, "annotations")      # optional
        annotations = to_list(annotations)
        modifiers = getattr(self, "modifiers")  # optional
        modifiers = to_list(modifiers)
        name = WrappedNode(getattr(self, "name"), "name")  # required
        arguments = getattr(self, "arguments")      # optional
        arguments = to_list(arguments)
        bodys = getattr(self, "body")       # optional
        bodys = to_list(bodys)

        result = ""
        if annotations:
            result += " ".join([annotation.to_code(show_plh=show_plh) for annotation in annotations])
        result += " "
        if modifiers:
            result += " ".join([modifier.to_code(show_plh=show_plh) for modifier in modifiers])
        result += name.to_code(show_plh=show_plh)
        result += "("
        if arguments:
            result += ",".join([argument.to_code(show_plh=show_plh) for argument in arguments])
        result += ")"
        if bodys:
            result += "{" + "".join([statement.to_code(show_plh=show_plh) for statement in bodys]) + "}"


class AnnotationMethod(Declaration):
    attrs = ("name", "return_type", "dimensions", "default")

