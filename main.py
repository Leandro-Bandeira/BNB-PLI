import gurobipy as gp
from gurobipy import GRB
import math


## Classe responsável por armazenar a solução dual/primal final ##
class Solution():

    def __init__(self, z_dual, z_primal):
        self.z_dual = z_dual
        self.z_primal = z_primal
        

## Class responsável por armazenar as configurações do modelo
class Problem():
    def __init__(self, count_vars, count_constr, coef_obj, coef_constr, new_constr=None):
        self.count_vars = count_vars
        self.count_constr = count_constr
        self.coef_constr = coef_constr
        self.coef_obj = coef_obj
        # New_constr é uma lista de tuplas da forma [(index, value)]
        # onde index indica o valor da variável que ocorreu branch e value o valor que será igualada
        self.new_constr = new_constr 

## Node representa um nó na árvore, no qual Problem é a configuração do modelo e gp.Model o modelo ##
class Node():
    def __init__(self, problem: Problem, model: gp.Model):
        self.problem = problem
        self.model = model

## Leitor de instancia ##
def read_instance(path: str):
    count_vars = 0
    count_constr = 0
    coef_obj = []
    coef_constr = []
    
    with open(path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                count_vars, count_constr = map(int, line.split())
            elif i == 1:
                coef_obj = list(map(int, line.split()))
            else:
                coef_constr.append(list(map(int, line.split())))

                

    return(
        count_vars,
        count_constr,
        coef_obj,
        coef_constr
    )

# Verifica se o modelo relaxado é viável para o problema original
# Se todas as variáveis são zero ou um.
def is_feasible(model):
    tolerance = 1e-6
    for var in model.getVars():
        if var.X is None:
            return False
        if not (abs(var.X - 0) <= tolerance or abs(var.X - 1) <= tolerance):
            return False
    return True

            
    
def solve_relax_problem(problem: Problem):

    count_vars = problem.count_vars
    count_constr = problem.count_constr
    coef_obj = problem.coef_obj
    coef_constr = problem.coef_constr
    new_constr = problem.new_constr

    model = gp.Model("Branch-And-Bound")
    model.setParam('OutputFlag', 0)
    # add variables
    #x = model.addVars(count_vars, vtype=GRB.BINARY, name='x')
    x = model.addVars(count_vars, lb = 0, ub=1, vtype=GRB.CONTINUOUS, name='x')

    # Define a função objetivo #
    model.setObjective(
        gp.quicksum(
            x[i] * coef_obj[i] for i in range(count_vars)
        ),
        GRB.MAXIMIZE
    )

    # Define as restrições originais #
    for i in range(count_constr):
        size_constr = len(coef_constr[i])
        model.addConstr(
            gp.quicksum(
                x[j] * coef_constr[i][j] for j in range(size_constr - 1)
            ) <= coef_constr[i][size_constr - 1]
        )

    # Define novas restrições #
    if new_constr:
        for (var_idx, val) in new_constr:
            model.addConstr(x[var_idx] == val)
    model.optimize()
    return model

def branching(node: Node):
    model = node.model
    problem = node.problem
    node_a = None
    node_b = None
    
    distances = [abs(var.X - 0.5) for var in model.getVars()]
    # Encontra o índice da menor distância
    closest_index = distances.index(min(distances))

    # Definindo valores do node_a
    # node_a: adiciona nova restrição a
    if problem.new_constr:
        new_constr_a = problem.new_constr + [(closest_index, 1)]
        new_constr_b = problem.new_constr + [(closest_index, 0)]
    else:
        new_constr_a = [[closest_index, 1]]
        new_constr_b = [[closest_index, 0]]

    problem_a = Problem(problem.count_vars, problem.count_constr, problem.coef_obj, problem.coef_constr, new_constr_a)

    # node_b: adiciona nova restrição a 0
    problem_b = Problem(problem.count_vars, problem.count_constr, problem.coef_obj, problem.coef_constr, new_constr_b)

    model_a = solve_relax_problem(problem_a)
    node_a = Node(problem_a, model_a)
    # Verifica se o novo modelo é viável
    if model_a.status == GRB.Status.INFEASIBLE:
        node_a = None
        
    model_b = solve_relax_problem(problem_b)
    node_b = Node(problem_b, model_b)
    # Verifica se o novo modelo é viável
    if model_b.status == GRB.Status.INFEASIBLE:
        node_b = None
    
    return (node_a, node_b)

def bnb(root: Node):
    stack = []
    stack.append(root)
    solution = Solution(math.inf, -math.inf)
    solution.z_dual = root.model.objVal
    
    while stack:
        current_node = stack[0]
        print(f"Current_node_FO: {current_node.model.objVal}")
        stack.pop(0)
        node_a, node_b = branching(current_node)
        
        # Se ambos forem podeados por inviabilidade
        if not node_a and not node_b:
            continue
        
        obj_val_a = -math.inf
        obj_val_b = -math.inf

        # Se o node a ou node b não foi podado por inviabilidade
        if node_a:
            obj_val_a = node_a.model.objVal
        if node_b:
            obj_val_b = node_b.model.objVal

        
        solution.z_dual = max([obj_val_a, obj_val_b]) 

        # Poda por limite, ou seja, se o valores de node a ou node b nao foram podados por inviabilidade
        # Mas se ocorrer da sua FO ser menor que o primal conhecido, podamos por limite
        if obj_val_a >= 0 and node_a.model.objVal <= solution.z_primal:
            node_a = None
        if obj_val_b >= 0 and node_b.model.objVal <= solution.z_primal:
            node_b = None

        # Se ambos forem podados por limite
        if not node_a and not node_b:
            continue
        
        new_nodes = [node_a, node_b]
        # Adiciona os nós, dos dois nós, aqueles que não foram podados, verifica se é viável para o problema original
        # Caso forem viáveis para o problema original, atualiza o z_primal
        # É garanttido que os nós restantes tenham objVal maior ou igual ao z_primal conhecido
        for node in new_nodes:
            if node:
                if is_feasible(node.model):
                    solution.z_primal = node.model.objVal

        print(f"Current z dual: {solution.z_dual}")
        print(f"Current z primal: {solution.z_primal}")
       
        # Adiciona na stack apenas aqueles que não foram podados
        stack += [node for node in new_nodes if node is not None]
        
        print(f"Stack size: {len(stack)}")
    return solution
        

                    
if __name__=="__main__":
    path_instance = "teste0.txt"

    count_vars, count_constr, coef_obj, coef_constr = read_instance(path_instance)

    ## Resolve o problema original relaxado ##
    root_problem = Problem(count_vars, count_constr, coef_obj, coef_constr)
    root_model = solve_relax_problem(root_problem)

    # Se ele for viável para o problema original, temos a solução ótima
    if is_feasible(root_model):
        print(root_model.objVal)
    else:
        # Caso não for viável para o problema original, criamos o BNB ##
        root = Node(root_problem, root_model)        
        sol = bnb(root)
        print(f"Final solution: {sol.z_primal}")
