from typing import Dict


class Node:
    
    def __init__(self, node_id: int, node_info: Dict = None):
        self.node_id = node_id
        self.node_info = node_info
        
    def __str__(self) -> str:
        return str(self.node_id) + \
                (": " + str(self.node_info) if self.node_info else "")
    
    def __repr__(self) -> str:
        return str(self.node_id) + \
                (": " + str(self.node_info) if self.node_info else "")
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Node):
            return self.node_id == __o.node_id
        return False
    
    def __hash__(self) -> int:
        return hash(self.node_id)