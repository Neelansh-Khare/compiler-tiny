from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    # Keywords
    MAIN = auto()
    VAR = auto()
    LET = auto()
    CALL = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    FI = auto()
    WHILE = auto()
    DO = auto()
    OD = auto()
    RETURN = auto()
    FUNCTION = auto()
    VOID = auto()
    
    # Symbols
    SEMICOLON = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    DOT = auto()
    ASSIGN = auto()  # <-
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    EQUALS = auto()      # ==
    NOTEQUALS = auto()   # !=
    LESS = auto()        # <
    LESSEQ = auto()      # <=
    GREATER = auto()     # >
    GREATEREQ = auto()   # >=
    
    # Other
    IDENTIFIER = auto()
    NUMBER = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.text[0] if text else None

    def error(self):
        raise Exception(f'Invalid character "{self.current_char}" at line {self.line}, column {self.column}')

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            if self.current_char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        result = ''
        token_column = self.column
        
        while self.current_char and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        
        return Token(TokenType.NUMBER, result, self.line, token_column)

    def identifier(self) -> Token:
        result = ''
        token_column = self.column
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        keywords = {
            'main': TokenType.MAIN,
            'var': TokenType.VAR,
            'let': TokenType.LET,
            'call': TokenType.CALL,
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'else': TokenType.ELSE,
            'fi': TokenType.FI,
            'while': TokenType.WHILE,
            'do': TokenType.DO,
            'od': TokenType.OD,
            'return': TokenType.RETURN,
            'function': TokenType.FUNCTION,
            'void': TokenType.VOID,
        }
        
        token_type = keywords.get(result.lower(), TokenType.IDENTIFIER)
        return Token(token_type, result, self.line, token_column)

    def get_next_token(self) -> Token:
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()

            # Handle symbols
            if self.current_char == '{':
                self.advance()
                return Token(TokenType.LBRACE, '{', self.line, self.column - 1)
            
            if self.current_char == '}':
                self.advance()
                return Token(TokenType.RBRACE, '}', self.line, self.column - 1)
            
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', self.line, self.column - 1)
            
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', self.line, self.column - 1)
            
            if self.current_char == ';':
                self.advance()
                return Token(TokenType.SEMICOLON, ';', self.line, self.column - 1)
            
            if self.current_char == '.':
                self.advance()
                return Token(TokenType.DOT, '.', self.line, self.column - 1)
            
            if self.current_char == ',':
                self.advance()
                return Token(TokenType.COMMA, ',', self.line, self.column - 1)

            # Handle operators
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS, '+', self.line, self.column - 1)
            
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS, '-', self.line, self.column - 1)
            
            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MULTIPLY, '*', self.line, self.column - 1)
            
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE, '/', self.line, self.column - 1)

            # Handle two-character operators
            if self.current_char == '<':
                self.advance()
                if self.current_char == '-':
                    self.advance()
                    return Token(TokenType.ASSIGN, '<-', self.line, self.column - 2)
                elif self.current_char == '=':
                    self.advance()
                    return Token(TokenType.LESSEQ, '<=', self.line, self.column - 2)
                return Token(TokenType.LESS, '<', self.line, self.column - 1)

            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.GREATEREQ, '>=', self.line, self.column - 2)
                return Token(TokenType.GREATER, '>', self.line, self.column - 1)

            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.EQUALS, '==', self.line, self.column - 2)
                self.error()

            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.NOTEQUALS, '!=', self.line, self.column - 2)
                self.error()

            self.error()

        return Token(TokenType.EOF, '', self.line, self.column)