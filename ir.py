from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from projectParser import Node, Number, BinOp, Variable, Assignment, VarDecl, FunctionCall, IfStatement, Program, WhileStatement
from lexer import TokenType

# IR (Intermediate Representation) Classes
class Instruction:
    def get_users(self) -> List[str]:
        return getattr(self, 'users', [])

@dataclass
class ConstInstruction:
    value: int
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = const #{self.value}"

@dataclass
class AddInstruction:
    left: str
    right: str
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = add {self.left} {self.right}"

@dataclass
class SubInstruction:
    left: str
    right: str
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = sub {self.left} {self.right}"

@dataclass
class MulInstruction:
    left: str
    right: str
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = mul {self.left} {self.right}"

@dataclass
class DivInstruction:
    left: str
    right: str
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = div {self.left} {self.right}"

@dataclass
class CompareInstruction:
    left: str
    right: str
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = cmp {self.left} {self.right}"

@dataclass
class BranchInstruction:
    condition: str
    true_target: int
    false_target: int
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"branch {self.condition} ? BB{self.true_target} : BB{self.false_target}"

@dataclass
class JumpInstruction:
    target: int
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"jump BB{self.target}"

@dataclass
class ReadInstruction:
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = read"

@dataclass
class WriteInstruction:
    value: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"write {self.value}"

@dataclass
class WriteNLInstruction:
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return "writeNL"

@dataclass
class PhiInstruction:
    values: List[str]
    result: str
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        values_str = ", ".join(self.values)
        return f"{self.result} = phi({values_str})"

@dataclass
class SetParamInstruction:
    param_num: int    # Which parameter (1, 2, 3, etc.)
    value: str       # SSA value to pass as parameter
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"setparam {self.param_num} {self.value}"

@dataclass
class CallInstruction:
    function: str    # Name of function to call
    result: str      # SSA variable to store result in
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = call {self.function}"

@dataclass
class GetParamInstruction:
    param_num: int    # Which parameter to get (1, 2, 3, etc.)
    result: str       # SSA variable to store parameter in
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.result} = getparam {self.param_num}"

@dataclass
class ReturnInstruction:
    value: Optional[str] = None  # Value to return (None for void functions)
    users: List[str] = field(default_factory=list)
    
    def __str__(self):
        if self.value is not None:
            return f"return {self.value}"
        return "return"
    
@dataclass
class FunctionDecl(Node):
    name: str                    # Function name
    params: List[str]            # Parameter names
    return_type: Optional[str]   # "void" or None for non-void functions
    var_decl: Optional[VarDecl]  # Local variable declarations
    body: List[Node]             # Function body statements

class BasicBlock:
    def __init__(self, id: int):
        self.id = id
        self.instructions: List[Instruction] = []
        self.predecessors: Set[int] = set()
        self.successors: Set[int] = set()
        self.dominators: Set[int] = set()
        self.dominated_by: Set[int] = set()
        self.idom: Optional[int] = None

    def add_instruction(self, instruction: Instruction) -> None:
        self.instructions.append(instruction)

    def __str__(self):
        header = f"BB{self.id}:"
        instr = '\n'.join(f"  {instruction}" for instruction in self.instructions)
        return f"{header}\n{instr}"

    def get_last_instruction(self) -> Optional[Instruction]:
        return self.instructions[-1] if self.instructions else None

    def is_terminated(self) -> bool:
        last = self.get_last_instruction()
        return isinstance(last, (JumpInstruction, BranchInstruction))
    
class SSAGenerator:
    def __init__(self):
        self.var_counter = 0
        self.inst_counter = 0  # For instruction ID counting
        self.block_counter = 0
        self.current_block: Optional[BasicBlock] = None
        self.blocks: List[BasicBlock] = []
        self.header_phis: Dict[int, Dict[str, str]] = {}  # Maps block ID to var->phi mappings
        self.loop_stack: List[int] = []  # Stack of loop header block IDs
        self.variable_values: Dict[str, str] = {}  # Track latest values
        self.symbol_table: Dict[str, str] = {}
        self.current_phi_values: Dict[str, str] = {}  # Add this for tracking phi results
        self.dominance_computed = False
        self.functions: Dict[str, List[BasicBlock]] = {"main": []}
        
        # Create initial block
        self.create_block()

    def create_block(self) -> BasicBlock:
        block = BasicBlock(self.block_counter)
        self.block_counter += 1
        self.blocks.append(block)
        self.current_block = block
        self.functions["main"] = self.blocks
        return block
    

    def get_current_value(self, var_name: str) -> str:
        """Get most recent SSA value for a variable"""
        if var_name in self.current_phi_values:
            return self.current_phi_values[var_name]
        return self.symbol_table.get(var_name, var_name)

    def new_var(self) -> str:
        self.var_counter += 1
        return f"%{self.var_counter}"

    def add_instruction(self, instruction: Instruction):
        self.inst_counter += 1
        instruction.id = self.inst_counter  # Add ID to each instruction
        self.current_block.instructions.append(instruction)

    def visit(self, node: Node) -> str:
        method = f'visit_{type(node).__name__}'
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node):
        raise Exception(f'No visit_{type(node).__name__} method')

    def visit_Number(self, node: Number) -> str:
        result = self.new_var()
        self.add_instruction(ConstInstruction(node.value, result))
        return result

    def visit_BinOp(self, node: BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        result = self.new_var()
        
        if node.op.type == TokenType.PLUS:
            self.add_instruction(AddInstruction(left, right, result))
        elif node.op.type == TokenType.MINUS:
            self.add_instruction(SubInstruction(left, right, result))
        elif node.op.type == TokenType.MULTIPLY:
            self.add_instruction(MulInstruction(left, right, result))
        elif node.op.type == TokenType.DIVIDE:
            self.add_instruction(DivInstruction(left, right, result))
        elif node.op.type in {TokenType.EQUALS, TokenType.NOTEQUALS, 
                            TokenType.LESS, TokenType.LESSEQ,
                            TokenType.GREATER, TokenType.GREATEREQ}:
            self.add_instruction(CompareInstruction(left, right, result))
        
        return result
    
    def visit_WriteInstruction(self, node: WriteInstruction) -> None:
        # Use most recent SSA value when writing
        value = self.symbol_table.get(node.value, node.value)
        self.add_instruction(WriteInstruction(value))

    def visit_Program(self, node: Program) -> None:
        if node.var_decl:
            self.visit(node.var_decl)
        for statement in node.statements:
            self.visit(statement)

    def visit_VarDecl(self, node: VarDecl) -> None:
        for var in node.variables:
            self.symbol_table[var] = self.new_var()
            # Initialize with zero
            self.add_instruction(ConstInstruction(0, self.symbol_table[var]))

    def visit_FunctionCall(self, node: FunctionCall) -> str:
    # Handle built-in functions first
        if node.name == "InputNum":
            result = self.new_var()
            self.add_instruction(ReadInstruction(result))
            return result
        elif node.name == "OutputNum":
            if node.args:
                value = self.visit(node.args[0])
                if isinstance(node.args[0], Variable) and node.args[0].name in self.symbol_table:
                    value = self.symbol_table[node.args[0].name]
                self.add_instruction(WriteInstruction(value))
            return self.new_var()  # Dummy return value
        elif node.name == "OutputNewLine":
            self.add_instruction(WriteNLInstruction())
            return self.new_var()  # Dummy return value
            
        # Handle user-defined functions
        # Process arguments
        arg_values = []
        for arg in node.args:
            arg_value = self.visit(arg)
            arg_values.append(arg_value)
            
        # Generate setpar instructions for each argument
        for i, arg_value in enumerate(arg_values, 1):
            self.add_instruction(SetParamInstruction(i, arg_value))
            
        # Generate call
        result = self.new_var()
        self.add_instruction(CallInstruction(node.name, result))
        return result

    def visit_IfStatement(self, node: IfStatement) -> None:
        condition = self.visit(node.condition)
        
        then_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        
        else_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        
        merge_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        
        # Add branch instruction to current block
        self.add_instruction(BranchInstruction(condition, then_block.id, else_block.id))
        
        # Process then block
        self.current_block = then_block
        for stmt in node.then_branch:
            self.visit(stmt)
        self.add_instruction(JumpInstruction(merge_block.id))
        
        # Process else block
        self.current_block = else_block
        if node.else_branch:
            for stmt in node.else_branch:
                self.visit(stmt)
        self.add_instruction(JumpInstruction(merge_block.id))
        
        # Continue with merge block
        self.current_block = merge_block
        self.blocks.extend([then_block, else_block, merge_block])

    def visit_WhileStatement(self, node: WhileStatement) -> None:
        print("\n=== Starting new While Statement ===")
        
        # Create blocks
        header_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        print(f"Created header block BB{header_block.id}")
        
        body_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        print(f"Created body block BB{body_block.id}")
        
        exit_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        print(f"Created exit block BB{exit_block.id}")
        
        # Jump to header
        self.current_block.instructions.append(JumpInstruction(header_block.id))
        prev_block = self.current_block
        print(f"Added jump from BB{prev_block.id} to BB{header_block.id}")
        
        # Save current variable values
        saved_values = self.variable_values.copy()
        self.loop_stack.append(header_block.id)
        print(f"\nCurrent symbol table: {self.symbol_table}")
        
        # Setup header block
        self.current_block = header_block
        modified_vars = self._find_modified_variables(node.body)
        print(f"Variables modified in loop: {modified_vars}")
        
        # Initialize header_phis for this block if not exists
        if header_block.id not in self.header_phis:
            self.header_phis[header_block.id] = {}
            
        # Create phi nodes for modified variables
        for var_name in modified_vars:
            if var_name in self.symbol_table:
                current_value = self.symbol_table[var_name]
                new_var = self.new_var()
                phi_inst = PhiInstruction([current_value], new_var)
                self.current_block.instructions.append(phi_inst)
                
                # Update tracking
                self.header_phis[header_block.id][var_name] = new_var
                self.symbol_table[var_name] = new_var
                self.current_phi_values[var_name] = new_var
        
        # Process condition
        condition = self.visit(node.condition)
        self.current_block.instructions.append(BranchInstruction(condition, body_block.id, exit_block.id))
        
        # Process loop body
        saved_phis = self.current_phi_values.copy()
        self.current_block = body_block
        for stmt in node.body:
            self.visit(stmt)
        
        # Update phi nodes with loop-carried values
        for var_name in modified_vars:
            if var_name in self.header_phis[header_block.id]:
                phi_var = self.header_phis[header_block.id][var_name]
                current_value = self.symbol_table[var_name]
                for inst in header_block.instructions:
                    if isinstance(inst, PhiInstruction) and inst.result == phi_var:
                        if current_value not in inst.values:
                            inst.values.append(current_value)
        
        # Jump back to header
        self.current_block.instructions.append(JumpInstruction(header_block.id))
        
        # Add blocks
        self.blocks.extend([header_block, body_block])
        
        # Setup exit block and add to blocks
        self.current_block = exit_block
        self.blocks.append(exit_block)
        
        # Pop loop context
        self.loop_stack.pop()
        
        # Restore saved values but keep final phi values
        self.variable_values = saved_values.copy()
        if header_block.id in self.header_phis:
            for var_name, phi_var in self.header_phis[header_block.id].items():
                self.variable_values[var_name] = phi_var
                self.symbol_table[var_name] = phi_var
                
        print("=== Ending While Statement ===\n")

    def visit_Assignment(self, node: Assignment) -> None:
        value = self.visit(node.value)
        self.symbol_table[node.name] = value
        self.variable_values[node.name] = value

    def visit_Variable(self, node: Variable) -> str:
        return self.get_current_value(node.name)

    def _find_modified_variables(self, statements: List[Node]) -> Set[str]:
        """Find all variables that are modified within a set of statements"""
        modified = set()
        for stmt in statements:
            if isinstance(stmt, Assignment):
                modified.add(stmt.name)
            elif isinstance(stmt, WhileStatement):
                modified.update(self._find_modified_variables(stmt.body))
            elif isinstance(stmt, IfStatement):
                modified.update(self._find_modified_variables(stmt.then_branch))
                if stmt.else_branch:
                    modified.update(self._find_modified_variables(stmt.else_branch))
        return modified
        
    def visit_FunctionDecl(self, node: FunctionDecl) -> None:
        # Create new function context
        prev_function = self.current_function
        self.current_function = node.name
        
        # Create entry block for function
        entry_block = BasicBlock(self.block_counter)
        self.block_counter += 1
        self.current_block = entry_block
        
        # Add getpar instructions for parameters
        for i, param in enumerate(node.params, 1):
            result = self.new_var()
            self.add_instruction(GetParamInstruction(i, result))
            self.symbol_table[param] = result
            
        # Process function body
        if node.var_decl:
            self.visit(node.var_decl)
        for stmt in node.body:
            self.visit(stmt)
            
        # Store function blocks
        self.functions[node.name] = self.blocks
        
        # Restore previous context
        self.current_function = prev_function

    def compute_dominance(self):
        """Compute dominance information for all blocks"""
        if self.dominance_computed:
            return
            
        # Initialize dominance sets
        all_blocks = set(range(len(self.blocks)))
        for block in self.blocks:
            block.dominators = all_blocks.copy()
            
        # Entry block only dominates itself
        self.blocks[0].dominators = {0}
        
        # Iterative computation of dominance
        changed = True
        while changed:
            changed = False
            for block in self.blocks[1:]:  # Skip entry block
                new_doms = {block.id}
                if block.predecessors:
                    pred_doms = [self.blocks[pred].dominators for pred in block.predecessors]
                    new_doms.update(*pred_doms)
                
                if new_doms != block.dominators:
                    block.dominators = new_doms
                    changed = True
                    
        self.dominance_computed = True

    def add_phi_functions(self):
        """Add phi functions at join points where necessary"""
        for block in self.blocks:
            if len(block.predecessors) > 1:
                # Find variables that need phi functions
                need_phi = set()
                for pred in block.predecessors:
                    pred_block = self.blocks[pred]
                    for inst in pred_block.instructions:
                        if isinstance(inst, (AddInstruction, SubInstruction, MulInstruction, DivInstruction)):
                            need_phi.add(inst.result)
                
                # Add phi instructions at start of block
                for var in need_phi:
                    values = [var] * len(block.predecessors)
                    phi = PhiInstruction(values, self.new_var())
                    block.instructions.insert(0, phi)

class IRInterpreter:
    def __init__(self, blocks: List[BasicBlock]):
        self.blocks = {block.id: block for block in blocks}
        self.values = {}  # Stores the values of SSA variables
        self.current_block_id = 0  # Start at BB0
        self.functions = {}  # Map function names to their BasicBlock lists
        self.current_function = "main"


    def get_value(self, var: str) -> int:
        if var.startswith('%'):
            return self.values.get(var, 0)  # Return 0 for uninitialized variables
        return int(var)

    def execute_block(self, block: BasicBlock) -> Optional[int]:
        for inst in block.instructions:
            if isinstance(inst, ReadInstruction):
                while True:
                    try:
                        value = input("Enter a number: ")
                        self.values[inst.result] = int(value)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
                    except KeyboardInterrupt:
                        print("\nProgram terminated by user.")
                        exit(1)
            elif isinstance(inst, WriteInstruction):
                print(self.get_value(inst.value), end='')
                print()
            elif isinstance(inst, WriteNLInstruction):
                print()
            elif isinstance(inst, BranchInstruction):
                condition = self.get_value(inst.condition)
                return inst.true_target if condition else inst.false_target
            elif isinstance(inst, JumpInstruction):
                return inst.target
            elif isinstance(inst, ConstInstruction):
                self.values[inst.result] = inst.value
            
            elif isinstance(inst, AddInstruction):
                self.values[inst.result] = self.get_value(inst.left) + self.get_value(inst.right)
            
            elif isinstance(inst, SubInstruction):
                self.values[inst.result] = self.get_value(inst.left) - self.get_value(inst.right)
            
            elif isinstance(inst, MulInstruction):
                self.values[inst.result] = self.get_value(inst.left) * self.get_value(inst.right)
            
            elif isinstance(inst, DivInstruction):
                self.values[inst.result] = self.get_value(inst.left) // self.get_value(inst.right)
            
            elif isinstance(inst, CompareInstruction):
                self.values[inst.result] = self.get_value(inst.left) < self.get_value(inst.right)
        return None

    def run(self):
        while self.current_block_id is not None:
            block = self.blocks[self.current_block_id]
            self.current_block_id = self.execute_block(block)