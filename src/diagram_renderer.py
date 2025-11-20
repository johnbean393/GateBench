import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import schemdraw
from schemdraw import logic
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set
import pyparsing as pp

@dataclass
class LogicNode:
    kind: str
    value: str = ''
    children: List['LogicNode'] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    rank: int = 0
    parent_count: int = 0
    id: int = field(default_factory=lambda: 0)  # Placeholder for unique ID logic

    def __hash__(self):
        return id(self)


class DiagramRenderer:

    def __init__(
        self, 
        save_directory: str = './questions'
    ):
        self.save_directory = save_directory
        self.gate_spacing_x = 8.0
        self.gate_spacing_y = 2.0
        self.input_positions: Dict[str, Tuple[float, float]] = {}
        self._memo: Dict[Tuple, LogicNode] = {}  # For DAG construction

    def _create_expression_parser(self):
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

    def _get_or_create_node(self, kind: str, children: List[LogicNode], value: str = '') -> LogicNode:
        """Get existing node from memo or create new one (DAG construction)."""
        # Create a unique key for the node based on its content and children
        child_ids = tuple(id(c) for c in children)
        key = (kind, value, child_ids)
        
        if key in self._memo:
            node = self._memo[key]
            node.parent_count += 1
            return node
            
        node = LogicNode(kind=kind, value=value, children=children)
        # For root nodes or first usage, parent_count is 0 or handled by caller. 
        # We increment parent_count only on reuse or initial link.
        # Actually, let's manage parent_count strictly.
        self._memo[key] = node
        return node

    def _build_dag(self, tree) -> LogicNode:
        """Convert pyparsing result into a LogicNode DAG."""
        if isinstance(tree, str):
            return self._get_or_create_node(kind='var', children=[], value=tree)

        if isinstance(tree, pp.ParseResults):
            tree = tree.asList()

        if len(tree) == 2 and isinstance(tree[0], str) and tree[0].lower() == 'not':
            child = self._build_dag(tree[1])
            child.parent_count += 1
            return self._get_or_create_node(kind='not', children=[child])

        if len(tree) >= 3 and isinstance(tree[1], str):
            left = self._build_dag(tree[0])
            right = self._build_dag(tree[2])
            left.parent_count += 1
            right.parent_count += 1
            return self._get_or_create_node(kind=tree[1].lower(), children=[left, right])

        raise ValueError(f"Unsupported expression structure: {tree}")

    def _assign_ranks(self, node: LogicNode):
        """Recursively assign rank (depth) to nodes."""
        if not node.children:
            node.rank = 0
            return

        max_child_rank = 0
        for child in node.children:
            # Ensure child rank is calculated
            # Note: In a DAG, we might visit nodes multiple times, 
            # but rank should be fixed based on inputs.
            # We want rank = max(child_ranks) + 1
            if child.rank == 0 and child.kind != 'var': 
                # If rank is 0 it might be unvisited or actually 0.
                # Recurse if not visited (simplified check)
                self._assign_ranks(child)
            max_child_rank = max(max_child_rank, child.rank)
        
        node.rank = max_child_rank + 1

    def _calculate_layout(self, root: LogicNode):
        """
        Assign final (x, y) coordinates using a rank-based layout.
        1. Assign ranks (X coordinates).
        2. Group nodes by rank.
        3. Sort nodes within rank by ideal Y (average of inputs).
        4. Assign Y coordinates with collision avoidance.
        """
        # 1. Assign ranks (X-axis)
        # We need to traverse the whole graph.
        # Since it's a DAG rooted at 'root', we can traverse.
        nodes = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if n in nodes: continue
            nodes.add(n)
            stack.extend(n.children)
            
        # Create a map of Rank -> List[Node]
        # First, calculate ranks bottom-up
        # A simple way for DAG: Rank = max(parent_ranks) + 1? No, max(child_ranks) + 1.
        # Input vars are rank 0.
        # We need to ensure all children are processed.
        
        # Topological sort or iterative update
        # Let's just do a depth-limited recursion or relaxation
        for _ in range(len(nodes)): # Worst case depth
            changed = False
            for node in nodes:
                if node.kind == 'var':
                    node.rank = 0
                    continue
                
                new_rank = max((c.rank for c in node.children), default=-1) + 1
                if new_rank != node.rank:
                    node.rank = new_rank
                    changed = True
            if not changed:
                break

        # Group by rank
        rank_groups = defaultdict(list)
        for node in nodes:
            rank_groups[node.rank].append(node)
            
        max_rank = max(rank_groups.keys()) if rank_groups else 0

        # 2. Assign positions rank by rank
        # Input positions (Rank 0) are fixed by self.input_positions
        for node in rank_groups[0]:
            if node.kind == 'var':
                pos = self.input_positions.get(node.value, (0, 0))
                node.x, node.y = pos

        # Process Ranks 1..N
        for r in range(1, max_rank + 1):
            group = rank_groups[r]
            
            # Calculate ideal Y for each node (average of children Y)
            for node in group:
                if node.children:
                    node.y = sum(c.y for c in node.children) / len(node.children)
                else:
                    node.y = 0
            
            # Sort by ideal Y to maintain relative ordering
            group.sort(key=lambda n: n.y)
            
            # Spread nodes to prevent overlap
            # Simple greedy placement
            if not group: continue
            
            # Start placing from the top-most ideal position, but respect spacing
            # We'll run a pass to push nodes apart
            for i in range(1, len(group)):
                prev_node = group[i-1]
                curr_node = group[i]
                min_y = prev_node.y - self.gate_spacing_y # schemdraw y grows upwards? 
                # Actually schemdraw default: positive y is up.
                # Our inputs are at 0, -2, -4 (going down).
                # So we should sort descending?
                # Let's stick to the input order.
                pass

            # Let's re-sort descending (top to bottom) since our inputs are 0, -2, -4
            group.sort(key=lambda n: n.y, reverse=True)
            
            for i in range(1, len(group)):
                upper_node = group[i-1]
                curr_node = group[i]
                
                # Ensure current node is at least spacing below upper node
                # y coordinates are negative/decreasing
                max_y = upper_node.y - self.gate_spacing_y
                if curr_node.y > max_y:
                    curr_node.y = max_y
            
            # Assign X coordinates
            for node in group:
                # X is simple: Rank * spacing + offset
                # Input bus is at x=1.
                node.x = 1.0 + (node.rank * self.gate_spacing_x)


    def _route_wire(self, d: schemdraw.Drawing, start: Tuple[float, float], end: Tuple[float, float]):
        """Draw an orthogonal connection between start and end points."""
        # Ensure coordinates are simple tuples
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        
        # Stagger vertical segments based on source Y to avoid overlaps
        # Heuristic: Shift mid_x slightly based on p1[1] (start Y)
        # p1[1] is usually negative (0, -2, -4...).
        offset = p1[1] * 0.2  
        mid_x = (p1[0] + p2[0]) / 2 + offset
        
        # Draw wire using distinct segments to ensure visibility
        d += logic.Line().at(p1).to((mid_x, p1[1]))
        d += logic.Line().at((mid_x, p1[1])).to((mid_x, p2[1]))
        d += logic.Line().at((mid_x, p2[1])).to(p2)

    def _draw_dag(self, root: LogicNode, d: schemdraw.Drawing):
        """Draw the DAG nodes and connections."""
        nodes = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if n in nodes: continue
            nodes.add(n)
            stack.extend(n.children)
            
        # Sort nodes by rank for drawing order (inputs first)
        sorted_nodes = sorted(list(nodes), key=lambda n: n.rank)
        
        for node in sorted_nodes:
            if node.kind == 'var':
                continue # Already drawn as bus
                
            xy = (node.x, node.y)
            
            if node.kind == 'not':
                gate = logic.Not().at(xy).anchor('out')
                d += gate
                # Store exact anchors for children to use
                node.output_pos = gate.out
                
                # Connect input
                child = node.children[0]
                start_pos = getattr(child, 'output_pos', (child.x, child.y))
                self._route_wire(d, start_pos, gate.in1)
                
            elif node.kind in ('and', 'or', 'xor', 'nand', 'nor'):
                gate_map = {
                    'and': logic.And,
                    'or': logic.Or,
                    'xor': logic.Xor,
                    'nand': logic.Nand,
                    'nor': logic.Nor
                }
                gate_cls = gate_map[node.kind]
                gate = gate_cls().at(xy).anchor('out')
                d += gate
                node.output_pos = gate.out
                
                # Connect inputs
                left = node.children[0]
                right = node.children[1]
                
                start_left = getattr(left, 'output_pos', (left.x, left.y))
                start_right = getattr(right, 'output_pos', (right.x, right.y))
                
                self._route_wire(d, start_left, gate.in1)
                self._route_wire(d, start_right, gate.in2)
            
            # Draw dot if fan-out > 1 (parent_count > 1)
            if node.parent_count > 1:
                # The dot should be at the output position of this node
                # which is where the wire starts splitting
                if hasattr(node, 'output_pos'):
                    d += logic.Dot().at(node.output_pos)
                else:
                    # Fallback for vars (handled separately usually, but good for safety)
                    d += logic.Dot().at((node.x, node.y))



    def _build_circuit_manually(self, expr: str) -> schemdraw.Drawing | None:
        """
        Manually build circuit with shared inputs, DAG reuse, and tidy layout.
        """
        self._memo = {} # Reset memo
        expr = expr.lower()
        variables = sorted(set(re.findall(r'\b[a-z]\b', expr)))
        if not variables:
            return None
        
        d = schemdraw.Drawing()
        d.config(unit=2.5, fontsize=14, lw=1.4, bgcolor='white')
        
        # Create shared input terminals
        input_positions: Dict[str, Tuple[float, float]] = {}
        y_spacing = 2.0
        for i, var in enumerate(variables):
            y_pos = -i * y_spacing
            d += logic.Dot().at((0, y_pos)).label(var.upper(), loc='left')
            d += logic.Line().at((0, y_pos)).tox(1)
            input_positions[var] = (1, y_pos)
        self.input_positions = input_positions
        
        try:
            parser = self._create_expression_parser()
            parsed_tree = parser.parseString(expr, parseAll=True)[0]
            
            # 1. Build DAG
            dag_root = self._build_dag(parsed_tree)
            
            # 2. Calculate Layout
            self._calculate_layout(dag_root)
            
            # 3. Draw
            self._draw_dag(dag_root, d)
            
            # Add labeled output
            output_pos = (dag_root.x, dag_root.y)
            d += logic.Line().at(output_pos).tox(output_pos[0] + 1)
            d += logic.Dot().label('Q', loc='right')
                
        except Exception as e:
            print(f"Error building circuit: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return d

    # Function to render a diagram and save it to the save directory
    def render(
        self, 
        question: str, 
        file_name: str
    ):
        # Try manual circuit building first
        manual_diagram = self._build_circuit_manually(question)
        
        if manual_diagram is not None:
            manual_diagram.save(f'{self.save_directory}/{file_name}.png', transparent=False)
        else:
            # Fall back to logicparse if manual building fails
            from schemdraw.parsing import logicparse
            diagram = logicparse(question, outlabel='Q')
            diagram.config(bgcolor='white')
            diagram.save(f'{self.save_directory}/{file_name}.png', transparent=False)

if __name__ == '__main__':
    # Load expressions from JSON file
    with open('questions/expressions.json', 'r') as f:
        expressions_data = json.load(f)
    # Loop through each expression
    for idx, item in enumerate(expressions_data, start=1):
        expression = item['expression']
        gate_count = item['gate_count']
        # Create directory for this gate_count if it doesn't exist
        output_dir = f'./questions/images/{gate_count}'
        os.makedirs(output_dir, exist_ok=True)
        # Create DiagramRenderer with the correct save directory
        diagram_renderer = DiagramRenderer(save_directory=output_dir)
        # Render and save
        file_name = f'question_{idx}'
        diagram_renderer.render(expression, file_name)
    # Print success message
    print("All questions rendered")