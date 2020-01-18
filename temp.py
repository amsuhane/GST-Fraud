import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_node("A", count=50)
G.add_node("B", count=100)
G.add_node("C", count=300)
G.add_edge("A", "B", weight=0.33)
G.add_edge("A", "C", weight=0.66)
G.add_edge("C", "B", weight=0.99)
plt.figure(figsize=(15,15))
pos = nx.spring_layout(G, k=0.1)
node_size = [d['count']*100 for (n,d) in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_color='cyan', alpha=1.0, node_size=10)
nx.draw_networkx_labels(G, pos, fontsize=14)
edge_width = [d['weight']*10 for (u,v,d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black', width=edge_width)
plt.axis('off')
plt.show()
