import pyparsing as pp
import itertools
import re

class ExpressionEvaluator:
    def __init__(self):
        pp.ParserElement.enablePackrat()
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create a parser for boolean expressions."""
        variable = pp.Word(pp.alphas, max=1)
        operator_not = pp.CaselessKeyword("not")
        operator_and = pp.CaselessKeyword("and")
        operator_or = pp.CaselessKeyword("or")
        operator_xor = pp.CaselessKeyword("xor")
        operator_nand = pp.CaselessKeyword("nand")
        operator_nor = pp.CaselessKeyword("nor")
        
        expr = pp.infixNotation(
            variable,
            [
                (operator_not, 1, pp.opAssoc.RIGHT),
                (operator_and | operator_nand, 2, pp.opAssoc.LEFT),
                (operator_xor, 2, pp.opAssoc.LEFT),
                (operator_or | operator_nor, 2, pp.opAssoc.LEFT),
            ]
        )
        return expr

    def _evaluate_node(self, node, context):
        """Recursively evaluate the parsed node."""
        if isinstance(node, str):
            if node in context:
                return context[node]
            return node 

        if isinstance(node, pp.ParseResults):
            node = node.asList()

        if isinstance(node, list):
            if len(node) == 1:
                return self._evaluate_node(node[0], context)

            if len(node) == 2 and node[0].lower() == 'not':
                return not self._evaluate_node(node[1], context)

            value = self._evaluate_node(node[0], context)
            
            i = 1
            while i < len(node):
                op = node[i].lower()
                right = self._evaluate_node(node[i+1], context)
                
                if op == 'and':
                    value = value and right
                elif op == 'or':
                    value = value or right
                elif op == 'xor':
                    value = value != right
                elif op == 'nand':
                    value = not (value and right)
                elif op == 'nor':
                    value = not (value or right)
                else:
                    raise ValueError(f"Unknown operator: {op}")
                i += 2
            return value
            
        raise ValueError(f"Unexpected node type: {type(node)}")

    def parse(self, expression: str):
        """Parse the expression string into a node structure."""
        parsed = self.parser.parseString(expression, parseAll=True)
        # Convert to list immediately to avoid repeated conversion overhead
        result = parsed[0]
        if isinstance(result, pp.ParseResults):
            return result.asList()
        return result

    def evaluate(self, expression_or_node, context: dict) -> bool:
        """Evaluate an expression string or parsed node with a given context."""
        if isinstance(expression_or_node, str):
            node = self.parse(expression_or_node)
        else:
            node = expression_or_node
            
        return self._evaluate_node(node, context)

    def get_variables(self, expression: str) -> set:
        """Extract unique variables from expression."""
        return set(re.findall(r'\b[a-zA-Z]\b', expression))

    def check_equivalence(self, expr1: str, expr2: str) -> bool:
        """
        Check if two expressions are equivalent by comparing their truth tables.
        Returns True if equivalent, False otherwise.
        """
        vars1 = self.get_variables(expr1)
        vars2 = self.get_variables(expr2)
        all_vars = sorted(list(vars1.union(vars2)))
        
        # Pre-parse expressions for performance
        try:
            node1 = self.parse(expr1)
            node2 = self.parse(expr2)
        except Exception as e:
            print(f"Error parsing expressions: {e}")
            return False
        
        # Iterate through all truth combinations
        for values in itertools.product([False, True], repeat=len(all_vars)):
            context = dict(zip(all_vars, values))
            try:
                val1 = self.evaluate(node1, context)
                val2 = self.evaluate(node2, context)
                if val1 != val2:
                    return False
            except Exception as e:
                print(f"Error evaluating expressions: {e}")
                return False
                
        return True

if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    
    # Example from user
    expr = "not (((not (A xor B)) and ((B nand C) or A)) xor ((not A) nand (B or C)))"
    print(f"Expression: {expr}")
    
    try:
        vars_ = evaluator.get_variables(expr)
        print(f"Variables: {sorted(vars_)}")
        
        print("\nTruth Table:")
        print(f"{' '.join(sorted(vars_))} | Result")
        print("-" * (len(vars_) * 2 + 9))
        
        all_vars = sorted(list(vars_))
        # Pre-parse
        node = evaluator.parse(expr)
        
        for values in itertools.product([False, True], repeat=len(all_vars)):
            context = dict(zip(all_vars, values))
            result = evaluator.evaluate(node, context)
            vals_str = " ".join("1" if v else "0" for v in values)
            res_str = "1" if result else "0"
            print(f"{vals_str} | {res_str}")
    except Exception as e:
        print(f"Error: {e}")
