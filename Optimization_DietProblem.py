# Question 15.2:  The Diet Problem

# Import necessary libraries
from pulp import *
import pandas as pd

# Import the dataset and prepare it for subsequent analysis
dataset = pd.read_excel('diet.xls')
data = dataset[0:64]
data = data.values.tolist()

# Establish list of foods in the dataset
foods = [x[0] for x in data]

# Set up dictionary containing the unit cost of the foods
cost = dict([(x[0], float(x[1])) for x in data])

# Establish dictionaries for the nutrient content of each food
calories = dict([(x[0], float(x[3])) for x in data])
cholesterol = dict([(x[0], float(x[4])) for x in data])
total_fat = dict([(x[0], float(x[5])) for x in data])
sodium = dict([(x[0], float(x[6])) for x in data])
carbs = dict([(x[0], float(x[7])) for x in data])
diet_fiber = dict([(x[0], float(x[8])) for x in data])
protein = dict([(x[0], float(x[9])) for x in data])
vit_A = dict([(x[0], float(x[10])) for x in data])
vit_C = dict([(x[0], float(x[11])) for x in data])
calcium = dict([(x[0], float(x[12])) for x in data])
iron = dict([(x[0], float(x[13])) for x in data])

# Set up lists for minimum and maximum requirements for each nutrient
headers = list(dataset.columns.values)[3:]
min_req = []
max_req = []
for header in headers:
    min_req.append(dataset[header][65])
    max_req.append(dataset[header][66])

# Solve problem with initial constraints

# Minimization problem to minimize overall cost
prob = LpProblem('Diet Optimization', LpMinimize)

# Establish variable for amount of each food.  Amount >= zero.
foodVars = LpVariable.dicts("Foods", foods, 0)

# Objective function is the overall cost of the foods selected
prob += lpSum([cost[f] * foodVars[f] for f in foods])

# Constraints: min and max values for all nutrients
prob += lpSum([calories[f] * foodVars[f] for f in foods]) >= min_req[0]
prob += lpSum([calories[f] * foodVars[f] for f in foods]) <= max_req[0]
prob += lpSum([cholesterol[f] * foodVars[f] for f in foods]) >= min_req[1]
prob += lpSum([cholesterol[f] * foodVars[f] for f in foods]) <= max_req[1]
prob += lpSum([total_fat[f] * foodVars[f] for f in foods]) >= min_req[2]
prob += lpSum([total_fat[f] * foodVars[f] for f in foods]) <= max_req[2]
prob += lpSum([sodium[f] * foodVars[f] for f in foods]) >= min_req[3]
prob += lpSum([sodium[f] * foodVars[f] for f in foods]) <= max_req[3]
prob += lpSum([carbs[f] * foodVars[f] for f in foods]) >= min_req[4]
prob += lpSum([carbs[f] * foodVars[f] for f in foods]) <= max_req[4]
prob += lpSum([diet_fiber[f] * foodVars[f] for f in foods]) >= min_req[5]
prob += lpSum([diet_fiber[f] * foodVars[f] for f in foods]) <= max_req[5]
prob += lpSum([protein[f] * foodVars[f] for f in foods]) >= min_req[6]
prob += lpSum([protein[f] * foodVars[f] for f in foods]) <= max_req[6]
prob += lpSum([vit_A[f] * foodVars[f] for f in foods]) >= min_req[7]
prob += lpSum([vit_A[f] * foodVars[f] for f in foods]) <= max_req[7]
prob += lpSum([vit_C[f] * foodVars[f] for f in foods]) >= min_req[8]
prob += lpSum([vit_C[f] * foodVars[f] for f in foods]) <= max_req[8]
prob += lpSum([calcium[f] * foodVars[f] for f in foods]) >= min_req[9]
prob += lpSum([calcium[f] * foodVars[f] for f in foods]) <= max_req[9]
prob += lpSum([iron[f] * foodVars[f] for f in foods]) >= min_req[10]
prob += lpSum([iron[f] * foodVars[f] for f in foods]) <= max_req[10]

# Solve the problem given variables, objective function, and constraints
prob.solve()

# Print out results for the foods selected
results = {}
for var in prob.variables():
    if var.varValue > 0 and "food_select" not in var.name:
        results[var] = round(var.varValue, 2)
results

# Additional constraints

# Create a binary variable to denote food selection
foodVars_selected = LpVariable.dicts("food_select", foods, 0, 1, LpBinary)

# Constraint: minimum 0.1 serving if food selected
for food in foods:
    prob += foodVars[food] >= 0.1 * foodVars_selected[food]
    # Any food that is chosen must have a binary variable of one
    prob += foodVars_selected[food] >= foodVars[food] * 0.0000001

# Solve the problem given variables, objective function, and constraints
prob.solve()

# Print out results for the foods selected
results = {}
for var in prob.variables():
    if var.varValue > 0 and "food_select" not in var.name:
        results[var] = round(var.varValue, 2)
results

# Constraint: cannot choose both frozen broccoli and raw celery
prob += foodVars_selected['Frozen Broccoli'] + foodVars_selected['Celery, Raw'] <= 1

# Solve the problem given variables, objective function, and constraints
prob.solve()

# Print out results for the foods selected
results = {}
for var in prob.variables():
    if var.varValue > 0 and "food_select" not in var.name:
        results[var] = round(var.varValue, 2)
results

# Constraint: must choose at least three "protein" items (meat, fish, poultry, eggs, etc.)
prob += foodVars_selected['Tofu'] + foodVars_selected['Roasted Chicken'] + \
    foodVars_selected['Poached Eggs'] + foodVars_selected['Scrambled Eggs'] + \
    foodVars_selected['Bologna,Turkey'] + foodVars_selected['Frankfurter, Beef'] + \
    foodVars_selected['Ham,Sliced,Extralean'] + foodVars_selected['Kielbasa,Prk'] + \
    foodVars_selected['Pizza W/Pepperoni'] + foodVars_selected['Taco'] + \
    foodVars_selected['Hamburger W/Toppings'] + foodVars_selected['Hotdog, Plain'] + \
    foodVars_selected['Peanut Butter'] + foodVars_selected['Pork'] + foodVars_selected['Sardines in Oil'] + \
    foodVars_selected['White Tuna in Water'] + foodVars_selected['Chicknoodl Soup'] + \
    foodVars_selected['Splt Pea&Hamsoup'] + foodVars_selected['Vegetbeef Soup'] + \
    foodVars_selected['Neweng Clamchwd'] + foodVars_selected['New E Clamchwd,W/Mlk'] + \
    foodVars_selected['Beanbacn Soup,W/Watr'] >= 3

# Solve the problem given variables, objective function, and constraints
prob.solve()

# Print out results for the foods selected
results = {}
for var in prob.variables():
    if var.varValue > 0 and "food_select" not in var.name:
        results[var] = round(var.varValue, 2)
results

