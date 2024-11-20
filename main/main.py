import pandas as pd 
import pprint
import matplotlib.pyplot as plt
import networkx as nx

def load_data_as_dataframes(file_path_stations: str, file_path_liaisons: str) -> tuple:
    """
    Charge les données des stations et des liaisons depuis des fichiers CSV et les retourne sous forme de DataFrames.

    Parameters:
    - file_path_stations (str): Chemin vers le fichier contenant les informations des stations.
    - file_path_liaisons (str): Chemin vers le fichier contenant les informations des liaisons.

    Returns:
    - tuple: (stations, liaisons)
        - stations (pd.DataFrame): DataFrame contenant les informations des stations.
        - liaisons (pd.DataFrame): DataFrame contenant les informations des liaisons.

    Raises:
    - FileNotFoundError: Si l'un des fichiers est introuvable.
    - ValueError: Si les colonnes requises sont absentes des fichiers.
    """
    try:
        # Lecture des fichiers
        stations = pd.read_csv(file_path_stations, sep=" ;", header=0, engine='python')
        liaisons = pd.read_csv(file_path_liaisons, sep=" ", header=0, engine='python')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Fichier non trouvé : {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur de parsing des fichiers : {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de la lecture des fichiers : {e}")

    # Vérification des colonnes requises
    required_columns_stations = {'num_sommet', 'nom_sommet', 'numéro_ligne', 'si_terminus', 'branchement'}
    required_columns_liaisons = {'num_sommet1', 'num_sommet2', 'temps_en_secondes'}

    if not required_columns_stations.issubset(stations.columns):
        raise ValueError(f"Les colonnes requises pour les stations sont manquantes. Requis : {required_columns_stations}")
    if not required_columns_liaisons.issubset(liaisons.columns):
        raise ValueError(f"Les colonnes requises pour les liaisons sont manquantes. Requis : {required_columns_liaisons}")

    return stations, liaisons


def build_graph_from_dataframes(stations: pd.DataFrame, liaisons: pd.DataFrame) -> dict:
    """
    Construit un graphe sous forme de dictionnaire d'adjacence à partir des DataFrames des stations et des liaisons.

    Parameters:
    - stations (pd.DataFrame): DataFrame contenant les informations des stations.
    - liaisons (pd.DataFrame): DataFrame contenant les informations des liaisons.

    Returns:
    - dict: Dictionnaire d'adjacence représentant le graphe.

    Raises:
    - ValueError: Si les DataFrames fournis sont vides ou mal structurés.
    """
    if stations.empty or liaisons.empty:
        raise ValueError("Les DataFrames des stations et des liaisons ne doivent pas être vides.")

    graph = {}

    # Ajout des stations comme noeuds
    for _, row in stations.iterrows():
        try:
            graph[row['num_sommet']] = {
                'nom_sommet': row['nom_sommet'],
                'numéro_ligne': row['numéro_ligne'],
                'si_terminus': row['si_terminus'],
                'branchement': row['branchement'],
                'voisins': []
            }
        except KeyError as e:
            raise ValueError(f"Données manquantes dans les informations des stations : {e}")

    # Ajout des liaisons entre les noeuds
    for _, row in liaisons.iterrows():
        try:
            if row['num_sommet1'] in graph:
                graph[row['num_sommet1']]['voisins'].append({
                    'num_sommet': row['num_sommet2'],
                    'temps_en_secondes': row['temps_en_secondes']
                })
            else:
                print(f"Sommet {row['num_sommet1']} introuvable dans les stations.")

            if row['num_sommet2'] in graph:
                graph[row['num_sommet2']]['voisins'].append({
                    'num_sommet': row['num_sommet1'],
                    'temps_en_secondes': row['temps_en_secondes']
                })
            else:
                print(f"Sommet {row['num_sommet2']} introuvable dans les stations.")

        except KeyError as e:
            raise ValueError(f"Données manquantes dans les informations des liaisons : {e}")
        except Exception as e:
            print(f"Erreur inattendue lors de l'ajout des liaisons : {e}")

    return graph


def display_graph(graph: dict, max_nodes: int = 5):
    """
    Affiche un aperçu limité des noeuds du graphe pour une inspection rapide.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - max_nodes (int): Nombre maximum de noeuds à afficher.
    """
    pp = pprint.PrettyPrinter(indent=2)
    print(f"Affichage limité du Graphe: {max_nodes} noeuds:")
    for i, (node, data) in enumerate(graph.items()):
        if i >= max_nodes:
            print("... (affichage limité)\n")
            break
        print(f"Sommet {node}:")
        pp.pprint(data)
        print("\n")


def is_graph_connected(graph: dict) -> tuple:
    """
    Vérifie si un graphe est connexe en utilisant un parcours en profondeur (DFS).
    Retourne un booléen indiquant si le graphe est connexe et, en cas de non-connexité,
    les sommets non atteignables.

    Parameters:
    - graph (dict): Dictionnaire d'adjacence représentant le graphe.
      Chaque clé est un sommet et la valeur est un dictionnaire contenant les voisins.

    Returns:
    - tuple: (bool, set)
        - bool: True si le graphe est connexe, False sinon.
        - set: Ensemble des sommets non atteignables si le graphe n'est pas connexe,
               ou un ensemble vide si le graphe est connexe.

    Raises:
    - ValueError: Si le graphe est vide.
    """
    if not graph:
        raise ValueError("Le graphe est vide. Un graphe vide ne peut pas être connexe.")

    # Choisir un sommet de départ arbitraire
    start_node = next(iter(graph))
    visited = set()  # Ensemble des sommets visités

    def dfs(node: int):
        """ Effectue un parcours en profondeur (DFS) à partir d'un sommet donné """
        visited.add(node)
        for neighbor in graph[node].get('voisins', []):
            neighbor_num = neighbor.get('num_sommet')
            if neighbor_num not in visited:
                dfs(neighbor_num)

    # Lancer le DFS à partir du sommet de départ
    dfs(start_node)

    # Identifier les sommets non atteignables
    all_nodes = set(graph.keys())
    non_visited = all_nodes - visited

    # Le graphe est connexe si tous les sommets ont été visités
    is_connected = len(non_visited) == 0

    return is_connected, non_visited


def find_non_terminal_nodes_with_few_neighbors(graph: dict) -> list:
    """
    Identifie tous les sommets non terminus du graphe ayant moins de deux voisins.

    Parameters:
    - graph (dict): Dictionnaire d'adjacence représentant le graphe.
      Chaque clé représente un sommet et sa valeur contient les informations associées,
      notamment les voisins et l'indicateur de terminus.

    Returns:
    - list: Liste des sommets non terminus ayant moins de deux voisins.
    """
    nodes_with_few_neighbors = []

    for node, data in graph.items():
        neighbors = data.get('voisins', [])  # Récupère les voisins du sommet (liste vide par défaut)
        is_terminal = data.get('si_terminus', False)  # Vérifie si le sommet est un terminus

        # Condition pour les sommets non terminus avec moins de deux voisins
        if not is_terminal and len(neighbors) < 2:
            nodes_with_few_neighbors.append(node)

    return nodes_with_few_neighbors


def find_shortest_path_bellman_ford(graph: dict, start_node: int, end_node: int) -> tuple:
    """
    Trouve le plus court chemin entre deux sommets dans un graphe pondéré
    en utilisant l'algorithme de Bellman-Ford.
    
    Cet algorithme peut gérer des poids négatifs, mais il ne fonctionne pas 
    si le graphe contient des cycles de poids négatifs.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe, où chaque clé est un sommet,
      et la valeur est un dictionnaire contenant les voisins et les poids des arêtes.
      Exemple:
      {
          1: {'voisins': [{'num_sommet': 2, 'temps_en_secondes': 10}, ...]},
          2: ...
      }
    - start_node (int): Sommet de départ.
    - end_node (int): Sommet d'arrivée.

    Returns:
    - tuple: (chemin, distances)
        - chemin (list): Liste des sommets constituant le plus court chemin, ou [] si aucun chemin n'est trouvé.
        - distances (dict): Dictionnaire des distances minimales de `start_node` à chaque sommet.
    
    Raises:
    - ValueError: Si le sommet de départ ou d'arrivée est absent du graphe.
    """

    # Vérification des sommets de départ et d'arrivée
    if start_node not in graph or end_node not in graph:
        raise ValueError(f"Le sommet de départ ({start_node}) ou d'arrivée ({end_node}) n'existe pas dans le graphe.")

    # Initialisation des distances et des prédécesseurs
    distances = {node: float('inf') for node in graph}  # Toutes les distances initialisées à l'infini
    predecessors = {node: None for node in graph}  # Aucun prédécesseur au départ
    distances[start_node] = 0  # La distance au sommet de départ est 0

    # Relaxation des arêtes pour |V| - 1 itérations
    for _ in range(len(graph) - 1):
        for current_node in graph:
            for neighbor in graph[current_node]['voisins']:
                neighbor_node = neighbor['num_sommet']
                travel_time = neighbor['temps_en_secondes']
                
                # Si une distance plus courte est trouvée, on met à jour
                if distances[current_node] + travel_time < distances[neighbor_node]:
                    distances[neighbor_node] = distances[current_node] + travel_time
                    predecessors[neighbor_node] = current_node

    # Détection des cycles de poids négatifs
    for current_node in graph:
        for neighbor in graph[current_node]['voisins']:
            neighbor_node = neighbor['num_sommet']
            travel_time = neighbor['temps_en_secondes']
            if distances[current_node] + travel_time < distances[neighbor_node]:
                raise ValueError("Le graphe contient un cycle de poids négatif. Bellman-Ford ne peut pas calculer un chemin fiable.")

    # Reconstruction du chemin le plus court
    shortest_path = []
    current_node = end_node
    while current_node is not None:
        shortest_path.insert(0, current_node)  # On ajoute au début pour construire le chemin à l'envers
        current_node = predecessors[current_node]

    # Si la distance vers le noeud final est infinie, il n'y a pas de chemin
    if distances[end_node] == float('inf'):
        print("Aucun chemin trouvé entre le sommet de départ et le sommet d'arrivée.")
        return [], distances

    return shortest_path, distances


def generate_travel_instructions(graph: dict, path: list, distances: dict) -> list:
    """
    Génère des instructions détaillées pour un trajet dans un graphe de transport en commun.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe. Chaque clé est un sommet, 
      et la valeur contient des informations comme les voisins, le nom du sommet, et le numéro de ligne.
    - path (list): Liste ordonnée des sommets représentant le chemin à parcourir.
    - distances (dict): Dictionnaire des distances calculées à partir du sommet de départ.

    Returns:
    - list: Liste des instructions textuelles pour guider l'utilisateur le long du chemin.
    
    Raises:
    - ValueError: Si le chemin est vide ou si les sommets dans le chemin ne sont pas présents dans le graphe.
    """

    if not path:
        raise ValueError("Le chemin fourni est vide.")
    if any(station not in graph for station in path):
        raise ValueError("Certains sommets du chemin ne sont pas présents dans le graphe.")

    instructions = []
    station_count = 0
    segment_time = 0

    # Vérifie et ajuste les extrémités du chemin pour éviter les doublons
    if graph[path[0]]['nom_sommet'] == graph[path[1]]['nom_sommet']:
        ref_station = path[1]
    else:
        ref_station = path[0]

    if graph[path[-1]]['nom_sommet'] == graph[path[-2]]['nom_sommet']:
        path = path[:-1]

    # Initialisation de la ligne actuelle
    current_line = graph[ref_station]['numéro_ligne']
    instructions.append(f"Vous êtes à {graph[ref_station]['nom_sommet']}.")

    path = path[1:]  # On saute la station de départ déjà référencée
    previous_station = ref_station
    first_segment = True

    # Construction des instructions
    for station in path:
        if current_line != graph[station]["numéro_ligne"]:
            # Si changement de ligne
            direction = find_closest_terminus(graph, previous_station, ref_station, current_line)
            if first_segment:
                instructions.append(
                    f"Prenez la ligne {current_line} direction {direction}, pour {station_count} stations, "
                    f"durée estimée {segment_time // 60} minutes."
                )
                first_segment = False
            else:
                instructions.append(
                    f"A {graph[ref_station]['nom_sommet']}, changez et prenez la ligne {current_line} direction {direction}, "
                    f"pour {station_count} stations, durée estimée {segment_time // 60} minutes."
                )

            # Réinitialisation des compteurs pour le nouveau segment
            ref_station = station
            current_line = graph[station]["numéro_ligne"]
            station_count = 0
            segment_time = 0
        else:
            # Si la ligne reste la même, on accumule les stations et le temps
            station_count += 1
            try:
                segment_time += next(
                    neighbor['temps_en_secondes']
                    for neighbor in graph[previous_station]['voisins']
                    if neighbor['num_sommet'] == station
                )
            except StopIteration:
                raise ValueError(
                    f"Aucune connexion trouvée entre {previous_station} et {station} dans le graphe."
                )

        previous_station = station

    # Dernier segment
    direction = find_closest_terminus(graph, previous_station, ref_station, current_line)
    instructions.append(
        f"A {graph[ref_station]['nom_sommet']}, continuez sur la ligne {current_line} direction {direction}, "
        f"pour {station_count} stations, durée estimée {segment_time // 60} minutes."
    )

    # Temps total et arrivée
    try:
        total_time = distances[path[-1]] // 60
        instructions.append(
            f"Vous devriez arriver à {graph[path[-1]]['nom_sommet']} en environ {total_time} minutes."
        )
    except KeyError:
        raise ValueError("Le sommet d'arrivée est introuvable dans le dictionnaire des distances.")

    return instructions


def find_closest_terminus(graph: dict, current_station: int, reference_station: int, line_number: int) -> str:
    """
    Trouve le terminus le plus proche sur une ligne donnée pour une station actuelle.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe. Chaque clé est un sommet, 
      et les valeurs contiennent des informations comme le nom, les voisins, 
      si c'est un terminus, et le numéro de ligne.
    - current_station (int): La station actuelle avant de changer de ligne.
    - reference_station (int): La première station prise sur la ligne donnée.
    - line_number (int): Le numéro de la ligne.

    Returns:
    - str: Le nom du terminus le plus proche.

    Raises:
    - ValueError: Si aucun terminus n'est trouvé pour la ligne donnée.
    """
    # Récupère tous les terminus de la ligne spécifiée
    terminus_candidates = [
        station for station, data in graph.items()
        if data['numéro_ligne'] == line_number and data.get('si_terminus', False)
    ]

    # Vérifie si la station actuelle est un terminus
    if current_station in terminus_candidates:
        return graph[current_station]['nom_sommet']

    # Si un seul terminus existe, le retourner directement
    if len(terminus_candidates) == 1:
        return graph[terminus_candidates[0]]['nom_sommet']

    # Recherche du terminus le plus proche
    closest_terminus = None
    min_distance_to_current = float('inf')

    for terminus in terminus_candidates:
        # Calcul des distances à partir de la station actuelle et de la référence
        distance_from_current = count_stations_between(graph, current_station, terminus, line_number)
        distance_from_reference = count_stations_between(graph, reference_station, terminus, line_number)

        # Sélection du terminus le plus proche de la station actuelle
        if distance_from_reference > distance_from_current and distance_from_current < min_distance_to_current:
            min_distance_to_current = distance_from_current
            closest_terminus = graph[terminus]['nom_sommet']

    if closest_terminus is None:
        raise ValueError(f"Aucun terminus valable trouvé pour la ligne {line_number}.")

    return closest_terminus


def count_stations_between(graph: dict, start_station: int, end_station: int, line_number: int) -> int:
    """
    Compte le nombre de stations entre deux sommets sur la même ligne.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - start_station (int): Station de départ.
    - end_station (int): Station d'arrivée.
    - line_number (int): Le numéro de la ligne.

    Returns:
    - int: Nombre de stations entre `start_station` et `end_station` sur la ligne spécifiée,
           ou float('inf') si aucun chemin n'est trouvé.
    """
    # Vérifie si les stations de départ et d'arrivée sont valides
    if start_station not in graph or end_station not in graph:
        raise ValueError(f"Les stations {start_station} ou {end_station} n'existent pas dans le graphe.")

    visited = set()  # Ensemble des stations visitées
    queue = [(start_station, 0)]  # (station actuelle, nombre de stations parcourues)

    # Parcours en largeur pour trouver le chemin le plus court
    while queue:
        current_station, count = queue.pop(0)

        # Retourne le nombre de stations si la destination est atteinte
        if current_station == end_station:
            return count

        visited.add(current_station)

        # Parcours des voisins
        for neighbor in graph[current_station]['voisins']:
            neighbor_station = neighbor['num_sommet']
            if neighbor_station not in visited and graph[neighbor_station]['numéro_ligne'] == line_number:
                queue.append((neighbor_station, count + 1))

    # Si aucun chemin n'est trouvé
    return float('inf')


def prim_mst_no_library(graph: dict) -> tuple:
    """
    Implémente l'algorithme de Prim pour trouver l'ACPM sans utiliser de bibliothèque externe (comme heapq).
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe avec le format suivant :
        {
            1: {'nom_sommet': 'Station A', 'voisins': [{'num_sommet': 2, 'temps_en_secondes': 10}, ...]},
            2: ...
        }

    Returns:
    - tuple: (mst_edges, total_weight)
        - mst_edges (list): Liste des arêtes dans l'ACPM sous forme de tuples (sommet1, sommet2, poids).
        - total_weight (int): Poids total de l'ACPM.
    """
    if not graph:
        raise ValueError("Le graphe est vide. Impossible de calculer l'ACPM.")

    # Choisir un sommet arbitraire comme point de départ
    start_node = next(iter(graph))
    visited = set()  # Ensemble des sommets visités
    mst_edges = []  # Liste des arêtes dans l'ACPM
    total_weight = 0  # Poids total de l'ACPM

    # File de priorité (simple liste)
    priority_queue = []

    # Ajouter les arêtes du sommet de départ dans la file de priorité
    for neighbor in graph[start_node]['voisins']:
        priority_queue.append((neighbor['temps_en_secondes'], start_node, neighbor['num_sommet']))

    visited.add(start_node)

    # Fonction pour trouver l'arête de poids minimal dans la file de priorité
    def extract_min(queue):
        """Retire et retourne l'élément avec le plus faible poids."""
        min_index = 0
        for i in range(1, len(queue)):
            if queue[i][0] < queue[min_index][0]:
                min_index = i
        return queue.pop(min_index)

    # Tant que la file de priorité n'est pas vide et que tous les sommets ne sont pas visités
    while priority_queue and len(visited) < len(graph):
        # Extraire l'arête de poids minimal
        weight, from_node, to_node = extract_min(priority_queue)

        if to_node not in visited:
            # Ajouter l'arête à l'ACPM
            visited.add(to_node)
            mst_edges.append((from_node, to_node, weight))
            total_weight += weight

            # Ajouter les nouvelles arêtes depuis ce sommet à la file de priorité
            for neighbor in graph[to_node]['voisins']:
                if neighbor['num_sommet'] not in visited:
                    priority_queue.append((neighbor['temps_en_secondes'], to_node, neighbor['num_sommet']))

    # Vérifier si le graphe est connexe
    if len(visited) != len(graph):
        raise ValueError("Le graphe n'est pas connexe. Impossible de calculer un ACPM.")

    return mst_edges, total_weight


def plot_mst(graph: dict, mst_edges: list):
    """
    Visualise l'Arbre Couvrant de Poids Minimum (ACPM) avec Matplotlib.

    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - mst_edges (list): Liste des arêtes de l'ACPM sous forme de tuples (sommet1, sommet2, poids).
    """
    # Créer un graphe avec NetworkX
    G = nx.Graph()

    # Ajouter les sommets
    for node, data in graph.items():
        G.add_node(node, label=data['nom_sommet'])

    # Ajouter les arêtes de l'ACPM
    for from_node, to_node, weight in mst_edges:
        G.add_edge(from_node, to_node, weight=weight)

    # Générer une disposition des sommets
    pos = nx.spring_layout(G)  # Positionnement automatique

    # Dessiner les sommets
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=2, edge_color='orange')

    # Ajouter les étiquettes des sommets
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="black")

    # Ajouter les poids des arêtes
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Afficher le graphe
    plt.title("Arbre Couvrant de Poids Minimum (ACPM)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Chemins vers les fichiers de données
    file_path_stations = r"asset/stations.txt"
    file_path_liaisons = r"asset/liaisons.txt"
    
    try:
        # Chargement des DataFrames
        stations, liaisons = load_data_as_dataframes(file_path_stations, file_path_liaisons)
        
        if stations is not None and liaisons is not None:
            # Affichage des premières lignes pour inspection
            print("Aperçu des données des stations :")
            print(stations.head(), "\n")
            print("Aperçu des données des liaisons :")
            print(liaisons.head(), "\n")
            
            # Création du graphe
            graph = build_graph_from_dataframes(stations, liaisons)
            print("Graphe généré avec succès.\n")
            display_graph(graph, 5)
            
            # Vérification de la connexité
            is_connected, non_visited_nodes = is_graph_connected(graph)
            if is_connected:
                print("Le graphe est connexe. Toutes les stations sont accessibles.\n")
            else:
                print(f"Le graphe n'est pas connexe. Sommets non atteignables : {non_visited_nodes}")
                print("Ajoutez les liaisons manquantes pour assurer la connexité.\n")
            
            # Identification des sommets non terminus avec moins de deux voisins
            few_neighbors_nodes = find_non_terminal_nodes_with_few_neighbors(graph)
            if few_neighbors_nodes:
                print(f"Sommets non terminus ayant moins de deux voisins ({len(few_neighbors_nodes)}) : {few_neighbors_nodes}\n")
            else:
                print("Tous les sommets non terminus ont au moins deux voisins.\n")
            
            # Calcul du plus court chemin
            print("=== Calcul du Plus Court Chemin ===")
            start_station_name = "Carrefour Pleyel"
            end_station_name = "Villejuif, P. Vaillant Couturier"
            
            # Trouver les IDs des stations dans le graphe
            start_station_id = next((k for k, v in graph.items() if v['nom_sommet'] == start_station_name), None)
            end_station_id = next((k for k, v in graph.items() if v['nom_sommet'] == end_station_name), None)
            
            if start_station_id is not None and end_station_id is not None:
                # Calcul du chemin avec Bellman-Ford
                shortest_path, distances = find_shortest_path_bellman_ford(graph, start_station_id, end_station_id)
                if shortest_path:
                    # Générer les instructions de voyage
                    instructions = generate_travel_instructions(graph, shortest_path, distances)
                    print("Instructions d'itinéraire :\n")
                    for step in instructions:
                        print(step)
                else:
                    print("Aucun chemin trouvé entre les stations spécifiées.\n")
            else:
                print(f"Erreur : Station de départ ou d'arrivée introuvable ({start_station_name}, {end_station_name}).\n")
            
            # === Ajout de l'ACPM ===
            print("\n=== Calcul de l'Arbre Couvrant de Poids Minimum (ACPM) ===")
            try:
                mst_edges, total_weight = prim_mst_no_library(graph)
                print("Arbre Couvrant de Poids Minimum (ACPM) :")
                print("Arêtes :")
                for edge in mst_edges:
                    print(f"  {edge[0]} → {edge[1]} (Poids : {edge[2]})")
                print(f"Poids total de l'ACPM : {total_weight}")
            
                # Appel de la fonction de visualisation existante
                plot_mst(graph, mst_edges)
            
            except ValueError as e:
                print(f"Erreur lors du calcul de l'ACPM : {e}")
            except ImportError as e:
                print("Erreur d'importation des bibliothèques nécessaires (Matplotlib ou NetworkX). Vérifiez votre installation.")
            
            
        else:
            print("Erreur : Les fichiers de données n'ont pas pu être correctement chargés.\n")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
