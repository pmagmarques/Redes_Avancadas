import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from io import StringIO
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix

def Pares_Atores_Frequentes(df):

    filmes_atores = df.groupby('Filme')['Ator'].apply(list).reset_index()
    
    # Encontrar pares de atores que participaram do mesmo filme
    pares_atores_comum = []
    
    for _, row in filmes_atores.iterrows():
        atores_comum = row['Ator']
        # Criar todos os pares de atores que participaram no mesmo filme
        if len(atores_comum) > 1:
            pares_atores_comum.extend(combinations(sorted(atores_comum), 2))
    
    # Contar a frequência de cada par de atores
    atores_comum_freq = pd.Series(pares_atores_comum).value_counts()
    
    # Exibir os pares de atores que mais participaram dos mesmos filmes
    return atores_comum_freq

def MatrizIncidencia():

    atores = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    filmes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    
    print(len(filmes))
    print(len(atores))
    
    print(filmes[:10])
    print(atores[:10])
    
    # Inicializando a matriz de incidência esparsa (LIL format, eficiente para inserção)
    incidence_matrix = lil_matrix((len(atores), len(filmes)), dtype=int)
    
    print (incidence_matrix)
    
    
    # Preenchendo a matriz de incidência com 1 onde houver uma aresta
    for i, ator in enumerate(atores):
        for j, filme in enumerate(filmes):
            if G.has_edge(ator, filme):
                incidence_matrix[i, j] = 1
    
    # Exibindo a matriz de incidência esparsa
    #print("Matriz de Incidência Esparsa:")
    #print(incidence_matrix)
    
    print("Fim")



def ActorListOfFilms(df):
    # Gives the list of films that each actor has partecipated
    ListOfFilms = df.groupby('Ator')['Filme'].apply(list).sort_values(key=lambda x: x.str.len(), ascending=False)
    
    print(ListOfFilms)

    print("Fim")

def co_actor_network(G):

    atores = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    print ("calculou")

    co_actor_net = nx.Graph()

    co_actor_net.clear()

    co_actor_net = nx.projected_graph(G,atores)
    
    return co_actor_net

def bipartite_netword(df):

    G = nx.Graph()

    G.clear()
    
    
    filmes = df['Filme'].unique()
    atores = df['Ator'].unique()
    

    G.add_nodes_from(atores, bipartite=0) 
    G.add_nodes_from(filmes, bipartite=1) 
    
    for _, row in df.iterrows():
        G.add_edge(row['Ator'],row['Filme'])
    
    return G

def printTop(N,Vector,VectorName):

    print("#####################################################################################")
    print(" ")

    print(VectorName + " Top"+str(N))
    for Node, Value in Vector[:N]:
        print(f"{Node}: {Value}")

def degree_centrality(G):

    degree_centrality_item = nx.degree_centrality(G)
    degree_centrality_ordered = sorted(degree_centrality_item.items(),key=lambda x: x[1], reverse=True) 

    return degree_centrality_ordered

def betweenness_centrality(G):

    betweenness_centrality_item = nx.betweenness_centrality(G)
    betweenness_centrality_ordered = sorted(betweenness_centrality_item.items(),key=lambda x: x[1], reverse=True) 

    return betweenness_centrality_ordered

def closeness_centrality(G):

    closeness_centrality_item = nx.closeness_centrality(G)
    closeness_centrality_ordered = sorted(closeness_centrality_item.items(),key=lambda x: x[1], reverse=True) 

    return closeness_centrality_ordered


def density(G):

    density_item = nx.density(G)
    ##density_ordered = sorted(density_item.items(),key=lambda x: x[1], reverse=True) 

    return density_item


def clustering(G):

    clustering_item = nx.clustering(G)
    clustering_ordered = sorted(clustering_item.items(),key=lambda x: x[1], reverse=True) 

    return clustering_ordered


def average_shortest_path_length(G):

    average_shortest_path_length_item = nx.average_shortest_path_length(G)
    average_shortest_path_length_ordered = sorted(average_shortest_path_length_item.items(),key=lambda x: x[1], reverse=True) 

    return average_shortest_path_length_ordered

def connected_components(G):

    connected_components_item = list(nx.connected_components(G))
 
    return connected_components_item


def assortativity(G):

    return nx.degree_assortativity_coefficient(G)




try:
    del df
    gc.collect()
except Exception as e:
    pass

try:
    G.clear()
except Exception as e:
    pass

# Carregar o dataset
##df = pd.read_csv(r'/Users/pedromarques/Library/CloudStorage/GoogleDrive-pmagmarques1971@gmail.com/My Drive/Master Degree/ISCTE/1º Semestre/Redes Avançadas/Projeto Final/ca-IMDB/ca-IMDB.edges',delim_whitespace=True,skiprows=3,names=['Filme', 'Ator'])

df = pd.read_csv(r'ca-IMDB.edges',delim_whitespace=True,skiprows=3,names=['Filme', 'Ator'])

### Análise exploratória dos Dados

## Numero de atores e filmes

print("Numero de Filme:" + str(df['Filme'].value_counts()))
print("Numero de Atores:" + str(df['Ator'].value_counts()))

## Filmes que um Ator já participou

print(df.groupby('Ator')['Filme'].apply(list).sort_values(key=lambda x: x.str.len(), ascending=False))

## Filmes e os atores que participaram

print(df.groupby('Filme')['Ator'].apply(list).sort_values(key=lambda x: x.str.len(), ascending=False))

print(Pares_Atores_Frequentes(df))

## Estatísticas descritivas
print(df.describe())



#5. Histograma das variáveis numéricas
#df.hist(figsize=(10, 8))
#plt.tight_layout()
#plt.show()

#6. Gráfico de dispersão entre duas variáveis
#sns.scatterplot(x='Filme', y='Ator', data=df)
#plt.show()

# 7. Matriz de correlação
#correlation_matrix = df.corr()
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.show()

##8. Gráfico de barras para variável categórica
#sns.countplot(x='Filmes', data=df)
#plt.show()

# 9. Boxplot para identificar outliers
#sns.boxplot(x=df['Atores'])
#plt.show()

#scaler = StandardScaler()
#df_scaled = scaler.fit_transform(df)

#pca = PCA(n_components=2)
#principal_components = pca.fit_transform(df_scaled)

# Visualizando os componentes principais
#print(principal_components)

G = bipartite_netword(df)

print("bipartite_netword")

print(G)

print("assortativity: " + str(assortativity(G)))

print("degree_centrality")

Var_degree_centrality = degree_centrality(G)

print(printTop(5,Var_degree_centrality,"degree_centrality"))

print("betweenness_centrality")

Var_betweenness_centrality = betweenness_centrality(G)

print(printTop(5,Var_betweenness_centrality,"betweenness_centrality"))

print("End")

print("closeness_centrality")

Var_closeness_centrality = closeness_centrality(G)

printTop(5,Var_closeness_centrality,"closeness_centrality")

print("End")

print("density")

Var_density = density(G)

print("density: " + str(Var_density))

print("clustering")

Var_clustering = clustering(G)

printTop(5,Var_clustering,"clustering")

print("End")

print("average_shortest_path_length")

Var_average_shortest_path_length = average_shortest_path_length(G)

printTop(5,Var_average_shortest_path_length,"average_shortest_path_length")

print("End")

print("connected_components")

Var_connected_components = connected_components(G)

printTop(5,Var_connected_components,"connected_components")

print("End")

print("co_actor_network")

Co_Atores = co_actor_network(G)

print(Co_Atores)

print("End")


Var_degree_centrality = degree_centrality(Co_Atores)

printTop(5,Var_degree_centrality,"degree_centrality")

Var_betweenness_centrality = betweenness_centrality(Co_Atores)

printTop(5,Var_betweenness_centrality,"betweenness_centrality")

Var_closeness_centrality = closeness_centrality(Co_Atores)

printTop(5,Var_closeness_centrality,"closeness_centrality")

Var_density = density(Co_Atores)

printTop(5,Var_density,"density")

Var_clustering = clustering(Co_Atores)

printTop(5,Var_clustering,"clustering")

Var_average_shortest_path_length = average_shortest_path_length(Co_Atores)

printTop(5,Var_average_shortest_path_length,"average_shortest_path_length")


Var_connected_components = connected_components(Co_Atores)

printTop(5,Var_connected_components,"connected_components")


print("assortativity: ")
print(assortativity(Co_Atores))




#nx.draw(G, with_labels=True, node_size=30, node_color='skyblue', font_size=12, font_weight='bold')

#plt.title("Bipartite Network of Movies and Actors")
#plt.show()
