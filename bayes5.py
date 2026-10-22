from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = DiscreteBayesianNetwork([
    ('Rain', 'WetGrass'),
    ('Sprinkler', 'WetGrass')
])

cpd_rain = TabularCPD('Rain', 2, [[0.8], [0.2]])
cpd_sprinkler = TabularCPD('Sprinkler', 2, [[0.6], [0.4]])

cpd_wet = TabularCPD(
    'WetGrass', 2,
    [[0.99, 0.10, 0.10, 0.01],
     [0.01, 0.90, 0.90, 0.99]],
    evidence=['Rain', 'Sprinkler'],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet)

print("Model valid:", model.check_model())

G = nx.DiGraph(model.edges())
nx.draw(G, with_labels=True, node_color='lightblue',
        node_size=2500, arrows=True)
plt.show()

infer = VariableElimination(model)
print(infer.query(['WetGrass']))