from dataclasses import dataclass
from typing import List, Optional, Set
from lexer import Token, TokenType, Lexer

@dataclass
class Node:
    pass

@dataclass
class Number(Node):
    value: int

@dataclass
class BinOp(Node):
    left: Node
    op: Token
    right: Node

@dataclass
class UnaryOp(Node):
    op: Token
    expr: Node

@dataclass
class Variable(Node):
    name: str

@dataclass
class Assignment(Node):
    name: str
    value: Node

@dataclass
class VarDecl(Node):
    variables: List[str]

@dataclass
class FunctionCall(Node):
    name: str
    args: List[Node]

@dataclass
class IfStatement(Node):
    condition: Node
    then_branch: List[Node]
    else_branch: Optional[List[Node]]

@dataclass
class WhileStatement(Node):
    condition: Node
    body: List[Node]

@dataclass
class ReturnStatement(Node):
    value: Optional[Node]

@dataclass
class Program(Node):
    var_decl: Optional[VarDecl]
    statements: List[Node]

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        self.variables: Set[str] = set()

    def error(self, message: str = 'Invalid syntax'):
        raise Exception(f'{message} at line {self.current_token.line}, column {self.current_token.column}')

    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f'Expected {token_type.name}, got {self.current_token.type.name}')

    def program(self) -> Program:
        """program : MAIN [varDecl] LBRACE statement_list RBRACE DOT"""
        self.eat(TokenType.MAIN)
        
        var_decl = None
        if self.current_token.type == TokenType.VAR:
            var_decl = self.var_decl()
            
        self.eat(TokenType.LBRACE)
        statements = self.statement_list()
        self.eat(TokenType.RBRACE)
        self.eat(TokenType.DOT)
        
        return Program(var_decl, statements)

    def var_decl(self) -> VarDecl:
        """varDecl : VAR ident { COMMA ident } SEMICOLON"""
        self.eat(TokenType.VAR)
        variables = []
        
        # First variable
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected identifier in variable declaration")
        
        variables.append(self.current_token.value)
        self.variables.add(self.current_token.value)
        self.eat(TokenType.IDENTIFIER)
        
        # Additional variables
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            if self.current_token.type != TokenType.IDENTIFIER:
                self.error("Expected identifier after comma in variable declaration")
            
            variables.append(self.current_token.value)
            self.variables.add(self.current_token.value)
            self.eat(TokenType.IDENTIFIER)
        
        self.eat(TokenType.SEMICOLON)
        return VarDecl(variables)

    def statement_list(self) -> List[Node]:
        """statement_list : statement (SEMICOLON statement)* [SEMICOLON]"""
        statements = [self.statement()]
        
        while self.current_token.type == TokenType.SEMICOLON:
            self.eat(TokenType.SEMICOLON)
            if self.current_token.type in {TokenType.LET, TokenType.CALL, TokenType.IF, TokenType.WHILE, TokenType.RETURN}:
                statements.append(self.statement())
        
        return statements

    def assignment(self) -> Assignment:
        """assignment : LET IDENTIFIER ASSIGN expr"""
        self.eat(TokenType.LET)
        
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected identifier in assignment")
        
        var_name = self.current_token.value
        if var_name not in self.variables:
            print(f"Warning: Variable '{var_name}' used before declaration")
        
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ASSIGN)
        expr = self.expr()
        return Assignment(var_name, expr)

    def function_call(self) -> FunctionCall:
        """functionCall : CALL IDENTIFIER [LPAREN [expr (COMMA expr)*] RPAREN]"""
        self.eat(TokenType.CALL)
        
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected function name")
        
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        
        args = []
        # Handle both forms: call f() and call f
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            if self.current_token.type != TokenType.RPAREN:
                args.append(self.expr())
                while self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
                    args.append(self.expr())
            self.eat(TokenType.RPAREN)
        
        return FunctionCall(func_name, args)

    def statement(self) -> Node:
        """
        statement : assignment
                 | functionCall
                 | ifStatement
                 | whileStatement
                 | returnStatement
        """
        if self.current_token.type == TokenType.LET:
            return self.assignment()
        elif self.current_token.type == TokenType.CALL:
            return self.function_call()
        elif self.current_token.type == TokenType.IF:
            return self.if_statement()
        elif self.current_token.type == TokenType.WHILE:
            return self.while_statement()
        elif self.current_token.type == TokenType.RETURN:
            return self.return_statement()
        else:
            self.error("Expected statement")

    def if_statement(self) -> IfStatement:
        """ifStatement : IF relation THEN statSequence [ELSE statSequence] FI"""
        self.eat(TokenType.IF)
        condition = self.relation()
        self.eat(TokenType.THEN)
        then_branch = self.statement_list()
        
        else_branch = None
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_branch = self.statement_list()
        
        self.eat(TokenType.FI)
        return IfStatement(condition, then_branch, else_branch)

    def while_statement(self) -> WhileStatement:
        """whileStatement : WHILE relation DO statSequence OD"""
        self.eat(TokenType.WHILE)
        condition = self.relation()
        self.eat(TokenType.DO)
        body = self.statement_list()
        self.eat(TokenType.OD)
        return WhileStatement(condition, body)

    def return_statement(self) -> ReturnStatement:
        """returnStatement : RETURN [expr]"""
        self.eat(TokenType.RETURN)
        value = None
        if self.current_token.type not in {TokenType.SEMICOLON, TokenType.RBRACE}:
            value = self.expr()
        return ReturnStatement(value)

    def relation(self) -> BinOp:
        """relation : expr (EQUALS | NOTEQUALS | LESS | LESSEQ | GREATER | GREATEREQ) expr"""
        left = self.expr()
        
        if self.current_token.type in {TokenType.EQUALS, TokenType.NOTEQUALS, 
                                     TokenType.LESS, TokenType.LESSEQ,
                                     TokenType.GREATER, TokenType.GREATEREQ}:
            op = self.current_token
            self.eat(self.current_token.type)
            right = self.expr()
            return BinOp(left=left, op=op, right=right)
        
        self.error("Expected relational operator")

    def expr(self) -> Node:
        """expression : term ((PLUS | MINUS) term)*"""
        node = self.term()
        
        while self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.term())
        
        return node

    def term(self) -> Node:
        """term : factor ((MULTIPLY | DIVIDE) factor)*"""
        node = self.factor()
        
        while self.current_token.type in {TokenType.MULTIPLY, TokenType.DIVIDE}:
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.factor())
        
        return node

    def factor(self) -> Node:
        """factor : IDENTIFIER | NUMBER | LPAREN expr RPAREN | functionCall"""
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Number(int(token.value))
        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            return Variable(token.value)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        elif token.type == TokenType.CALL: 
            return self.function_call()
        else:
            self.error("Expected factor")