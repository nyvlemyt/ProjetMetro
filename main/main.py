import pandas as pd 
import pprint

def create_dataframes(file_path_stations: str, file_path_liaisons: str):
    """
    Lit les fichiers de données pour les stations et les liaisons et renvoie deux DataFrames.
    
    Parameters:
    - file_path_stations (str): Chemin vers le fichier contenant les informations des stations.
    - file_path_liaisons (str): Chemin vers le fichier contenant les informations des liaisons.
    
    Returns:
    - tuple: DataFrames pour les stations et les liaisons.
    """
    try:
        # Lecture des fichiers avec pandas
        stations = pd.read_csv(file_path_stations, sep=" ;", header=0, engine='python')
        liaisons = pd.read_csv(file_path_liaisons, sep=" ", header=0, engine='python')
    except FileNotFoundError as e:
        print(f"Erreur : fichier non trouvé - {e}")
        return None, None
    except pd.errors.ParserError as e:
        print(f"Erreur de parsing : {e}")
        return None, None
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
        return None, None
    
    # Vérifie que les colonnes nécessaires sont présentes
    required_columns_stations = {'num_sommet', 'nom_sommet', 'numéro_ligne', 'si_terminus','branchement'}
    required_columns_liaisons = {'num_sommet1', 'num_sommet2', 'temps_en_secondes'}

    if not required_columns_stations.issubset(stations.columns):
        print(f"Erreur : Les colonnes des stations sont incomplètes.")
        return None, None
    if not required_columns_liaisons.issubset(liaisons.columns):
        print(f"Erreur : Les colonnes des liaisons sont incomplètes.")
        return None, None

    return stations, liaisons

def create_graphe(stations: pd.DataFrame, liaisons: pd.DataFrame) -> dict:
    """
    Crée un graphe sous forme de dictionnaire d'adjacence à partir des DataFrames des stations et des liaisons.

    Parameters:
    - stations (pd.DataFrame): DataFrame contenant les informations des stations.
    - liaisons (pd.DataFrame): DataFrame contenant les informations des liaisons.

    Returns:
    - dict: Dictionnaire d'adjacence représentant le graphe.
    """
    graphe = {}

    # Construction des noeuds
    for _, row in stations.iterrows():
        try:
            graphe[row['num_sommet']] = {
                'nom_sommet': row['nom_sommet'],
                'numéro_ligne': row['numéro_ligne'],
                'si_terminus': row['si_terminus'],
                'branchement': row['branchement'],
                'voisins': []
            }
        except KeyError as e:
            print(f"Erreur : clé manquante dans les données de station - {e}")
    
    # Ajout des liaisons aux deux sommets de chaque arc
    for _, row in liaisons.iterrows():
        try:
            # Ajout de num_sommet2 comme voisin de num_sommet1
            if row['num_sommet1'] in graphe:
                graphe[row['num_sommet1']]['voisins'].append({
                    'num_sommet': row['num_sommet2'],
                    'temps_en_secondes': row['temps_en_secondes']
                })
            else:
                print(f"Attention : sommet {row['num_sommet1']} n'existe pas dans les stations.")

            # Ajout de num_sommet1 comme voisin de num_sommet2
            if row['num_sommet2'] in graphe:
                graphe[row['num_sommet2']]['voisins'].append({
                    'num_sommet': row['num_sommet1'],
                    'temps_en_secondes': row['temps_en_secondes']
                })
            else:
                print(f"Attention : sommet {row['num_sommet2']} n'existe pas dans les stations.")

        except KeyError as e:
            print(f"Erreur : clé manquante dans les données de liaison - {e}")
        except Exception as e:
            print(f"Une erreur inattendue s'est produite lors de l'ajout des liaisons : {e}")
    
    return graphe

# Fonction pour visualiser le graphe de manière plus lisible
def afficher_graphe(graph: dict, max_sommets=5):
    """
    Affiche les premiers sommets et leurs voisins dans le graphe pour un aperçu rapide.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - max_sommets (int): Nombre maximum de sommets à afficher.
    """
    pp = pprint.PrettyPrinter(indent=2)
    print("Graphe (affichage limité):")
    for i, (sommet, data) in enumerate(graph.items()):
        if i >= max_sommets:
            print("... (affichage limité)")
            break
        print(f"Sommet {sommet}:")
        pp.pprint(data)
        print("\n")

def est_connexe(graph: dict) -> tuple:
    """
    Vérifie si un graphe est connexe en utilisant DFS et retourne les sommets non atteignables si le graphe n'est pas connexe.
    
    Parameters:
    - graph (dict): Dictionnaire d'adjacence représentant le graphe.
    
    Returns:
    - tuple: (bool, set) où le booléen indique si le graphe est connexe et l'ensemble contient les sommets non atteignables si le graphe n'est pas connexe.
    """
    if not graph:
        return False, set()  # Un graphe vide n'est pas connexe et n'a pas de sommets

    # Choisir un sommet de départ arbitraire
    start_node = next(iter(graph))
    visited = set()
    
    def dfs(node):
        """ Parcours en profondeur récursif """
        visited.add(node)
        #print(f"Visiting node {node}")
        #print(f"Visited nodes: {visited}")
        for voisin in graph[node]['voisins']:
            voisin_num = voisin['num_sommet']
            if voisin_num not in visited:
                dfs(voisin_num)

    # Lancer le DFS à partir du sommet de départ
    dfs(start_node)
    
    # Obtenir les sommets non atteignables
    all_nodes = set(graph.keys())
    non_visited = all_nodes - visited

    # Vérifier la connexité
    is_connected = len(non_visited) == 0
    
    return is_connected, non_visited

def trouver_sommets_non_terminus_avec_moins_de_deux_voisins(graph: dict) -> list:
    """
    Identifie tous les sommets non terminus du graphe qui ont moins de deux voisins.
    
    Parameters:
    - graph (dict): Dictionnaire d'adjacence représentant le graphe.
    
    Returns:
    - list: Liste des sommets non terminus ayant moins de deux voisins.
    """
    sommets_avec_moins_de_deux_voisins = []

    for sommet, data in graph.items():
        # Vérifie si le sommet n'est pas un terminus et a moins de deux voisins
        if not data.get('si_terminus', False) and len(data.get('voisins', [])) < 2:
            sommets_avec_moins_de_deux_voisins.append(sommet)
        # Vérifie si le sommet est un terminus et a moins de 1 voisins
        if not data.get('si_terminus', True) and len(data.get('voisins', [])) < 1:
            sommets_avec_moins_de_deux_voisins.append(sommet)
    
    return sommets_avec_moins_de_deux_voisins

def bellman_ford(graph: dict, start: int, end: int) -> list:
    """
    Calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - start (int): Sommet de départ.
    - end (int): Sommet d'arrivée.
    
    Returns:
    - list: Le chemin du plus court chemin avec les instructions.
    """
    # Initialisation des distances
    distances = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    distances[start] = 0

    # Relaxation des arêtes
    for _ in range(len(graph) - 1):
        for node in graph:
            for voisin in graph[node]['voisins']:
                voisin_num = voisin['num_sommet']
                temps = voisin['temps_en_secondes']
                if distances[node] + temps < distances[voisin_num]:
                    distances[voisin_num] = distances[node] + temps
                    predecessors[voisin_num] = node

    # Reconstruction du chemin
    chemin = []
    current = end
    while current is not None:
        chemin.insert(0, current)
        current = predecessors[current]
    
    if distances[end] == float('inf'):
        print("Aucun chemin trouvé.")
        return []
    
    print(f"Le chemin le plus court est {distances[end]} secondes.")
    print(f"Chemin : {chemin}")
    print(f"Prédécesseurs : {predecessors}")
    print(f"Distances : {distances}")
    return generate_instructions(graph, chemin, distances)

# def generate_instructions(graph: dict, chemin: list, distances: dict) -> list:
#     """
#     Génère les instructions pour le chemin donné.
    
#     Parameters:
#     - graph (dict): Dictionnaire représentant le graphe.
#     - chemin (list): Liste des sommets du chemin calculé.
#     - distances (dict): Dictionnaire des distances depuis le sommet de départ.
    
#     Returns:
#     - list: Instructions textuelles du chemin.
#     """
#     instructions = []
#     current_line = None
#     station_count = 0
#     segment_time = 0
    
#     for i in range(len(chemin) - 1):
#         current = chemin[i]
#         next_station = chemin[i + 1]
        
#         current_station = graph[current]
#         next_station_data = graph[next_station]
        
#         # Première station
#         if i == 0:
#             instructions.append(f"Vous êtes à {current_station['nom_sommet']}.")

#         # Changement de ligne ou continuation sur la même ligne
#         if current_station['numéro_ligne'] != next_station_data['numéro_ligne']:
#             # Si on change de ligne, on ajoute les informations du segment précédent
#             if current_line is not None:
#                 instructions.append(
#                     f"Prenez la ligne {current_line} pour {station_count} stations, "
#                     f"durée estimée {segment_time // 60} minutes."
#                 )
            
#             # Trouver la direction de la nouvelle ligne
#             direction_station = trouver_terminus_proche(graph, next_station, next_station_data['numéro_ligne'], distances)
            
#             # Ajouter l'instruction de changement de ligne
#             instructions.append(
#                 f"A {current_station['nom_sommet']}, changez et prenez la ligne "
#                 f"{next_station_data['numéro_ligne']} direction {direction_station}."
#             )
            
#             # Réinitialiser les compteurs pour le nouveau segment de ligne
#             current_line = next_station_data['numéro_ligne']
#             station_count = 1
#             segment_time = next(voisin['temps_en_secondes'] for voisin in current_station['voisins'] if voisin['num_sommet'] == next_station)
#         else:
#             # Si on reste sur la même ligne, on incrémente les compteurs
#             if current_line is None:
#                 current_line = current_station['numéro_ligne']
#             station_count += 1
#             segment_time += next(voisin['temps_en_secondes'] for voisin in current_station['voisins'] if voisin['num_sommet'] == next_station)
    
#     # Ajouter les informations pour le dernier segment de ligne
#     instructions.append(
#         f"Prenez la ligne {current_line} pour {station_count} stations, durée estimée {segment_time // 60} minutes."
#     )
    
#     # Temps total pour le trajet complet
#     instructions.append(f"Vous devriez arriver à {graph[chemin[-1]]['nom_sommet']} dans environ {distances[chemin[-1]] // 60} minutes.")
    
#     return instructions

# def trouver_terminus_proche(graph: dict, station_actuelle: int, ligne: int, distances: dict) -> str:
#     """
#     Trouve le terminus le plus proche sur une ligne donnée pour une station actuelle.
    
#     Parameters:
#     - graph (dict): Dictionnaire représentant le graphe.
#     - station_actuelle (int): Le sommet actuel.
#     - ligne (int): Le numéro de la ligne.
#     - distances (dict): Dictionnaire des distances calculées pour chaque sommet.
    
#     Returns:
#     - str: Le nom du terminus le plus proche.
#     """
#     terminus_candidates = [
#         sommet for sommet, data in graph.items()
#         if data['numéro_ligne'] == ligne and data['si_terminus'] is True
#     ]
    
#     # Trouver le terminus le plus proche
#     min_distance = float('inf')
#     closest_terminus = None
    
#     for terminus in terminus_candidates:
#         # Calculer la distance depuis la station actuelle jusqu'au terminus
#         distance_to_terminus = abs(distances[terminus] - distances[station_actuelle])
#         if distance_to_terminus < min_distance:
#             min_distance = distance_to_terminus
#             closest_terminus = graph[terminus]['nom_sommet']
    
#     return closest_terminus

def generate_instructions(graph: dict, chemin: list, distances: dict) -> list:
    """
    Génère les instructions pour le chemin donné.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - chemin (list): Liste des sommets du chemin calculé.
    - distances (dict): Dictionnaire des distances depuis le sommet de départ.
    
    Returns:
    - list: Instructions textuelles du chemin.
    """
    instructions = []
    current_line = None
    station_count = 0
    segment_time = 0
    first = True 
    nom = [graph[chemin[i]]['nom_sommet'] for i in range(len(chemin))]
    print("nom : ", nom)
    for i in range(len(chemin) - 1):
        current = chemin[i]
        next_station = chemin[i + 1]
        
        current_station = graph[current]
        next_station_data = graph[next_station]
        
        # Première station
        if i == 0:
            instructions.append(f"Vous êtes à {current_station['nom_sommet']}.")
            ref_station = current
        if current_station["nom_sommet"] == "Champs Élysées, Clémenceau": 
            print(i)
        # Changement de ligne ou continuation sur la même ligne
        if current_station['numéro_ligne'] != next_station_data['numéro_ligne']:
            # Si on change de ligne, on ajoute les informations du segment précédent
            if current_line is not None and first:
                #terminus_direction = find_correct_direction(graph, current, next_station, current_line, chemin[-1])
                terminus_direction = find_correct_direction2(graph, current, ref_station, current_line)
                instructions.append(
                    f"Prenez la ligne {current_line} direction {terminus_direction} pour {station_count} stations, "
                    f"durée estimée {segment_time // 60} minutes."
                )
                first = False
            elif current_line is not None :  
                terminus_direction = find_correct_direction2(graph, current, ref_station, current_line)
                # Ajouter l'instruction de changement de ligne
                instructions.append(
                    f"A {graph[ref_station]['nom_sommet']}, changez et prenez la ligne "
                    f"{current_line} direction {terminus_direction} pour {station_count} stations, durée estimée {segment_time // 60} minutes."
                )
            # Trouver la direction de la nouvelle ligne
            #terminus_direction = find_correct_direction(graph, next_station, chemin[-1], next_station_data['numéro_ligne'], chemin[-1])
            #terminus_direction = find_correct_direction2(graph, next_station, chemin[-1], next_station_data['numéro_ligne'])
            
            # Ajouter l'instruction de changement de ligne
            #instructions.append(
            #    f"A {current_station['nom_sommet']}, changez et prenez la ligne "
            #    f"{next_station_data['numéro_ligne']} direction {terminus_direction} pour {station_count} stations, durée estimée {segment_time // 60} minutes."
            #)
            
            # Réinitialiser les compteurs pour le nouveau segment de ligne
            ref_station = current
            current_line = next_station_data['numéro_ligne']
            station_count = 1
            segment_time = next(voisin['temps_en_secondes'] for voisin in current_station['voisins'] if voisin['num_sommet'] == next_station)
            print("segment_time : ", segment_time)
        else:
            # Si on reste sur la même ligne, on incrémente les compteurs
            if current_line is None:
                current_line = current_station['numéro_ligne']
            station_count += 1
            segment_time += next(voisin['temps_en_secondes'] for voisin in current_station['voisins'] if voisin['num_sommet'] == next_station)
    
    # Ajouter les informations pour le dernier segment de ligne
    terminus_direction = find_correct_direction(graph, chemin[-2], chemin[-1], current_line, chemin[-1])
    instructions.append(
        f"Prenez la ligne {current_line} direction {terminus_direction} pour {station_count} stations, durée estimée {segment_time // 60} minutes."
    )
    
    # Temps total pour le trajet complet
    instructions.append(f"Vous devriez arriver à {graph[chemin[-1]]['nom_sommet']} en environ {distances[chemin[-1]] // 60} minutes.")
    
    return instructions

def find_correct_direction(graph: dict, current_station: int, destination: int, ligne: int, terminus_target: int) -> str:
    """
    Trouve le terminus le plus proche sur une ligne donnée pour une station actuelle.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - current_station (int): Le sommet actuel.
    - destination (int): La station de destination.
    - ligne (int): Le numéro de la ligne.
    - terminus_target (int): Le terminus cible pour la direction.

    Returns:
    - str: Le nom du terminus le plus proche.
    """
    terminus_candidates = [
        sommet for sommet, data in graph.items()
        if data['numéro_ligne'] == ligne and data['si_terminus'] is True #and data['branchement'] == graph[destination]['branchement']
    ]
    if len (terminus_candidates) == 1:
        return graph[terminus_candidates[0]]['nom_sommet']

    min_distance_to_destination = float('inf')
    closest_terminus = None
    print("current_station : ", current_station)
    print("destination : ", destination)
    print("ligne : ", ligne)
    print("terminus_target : ", terminus_target)
    print('terminus_candidates : ', terminus_candidates)
    for terminus in terminus_candidates:
        print("terminus : ", terminus) 
        distance_from_current = count_stations_between(graph, current_station, terminus, ligne)
        print("distance_from_current : ", distance_from_current)
        distance_from_destination = count_stations_between(graph, destination, terminus, ligne)
        print("distance_from_destination : ", distance_from_destination)
        print("distance_from_destination < distance_from_current : ", distance_from_destination < distance_from_current)
        print("distance_from_destination < min_distance_to_destination : ", distance_from_destination < min_distance_to_destination)
        # Vérifier si le terminus est plus proche de la destination que de la station actuelle
        if distance_from_destination < distance_from_current and distance_from_destination < min_distance_to_destination:
            min_distance_to_destination = distance_from_destination
            print("min_distance_to_destination : ", min_distance_to_destination)
            closest_terminus = graph[terminus]['nom_sommet']
            print("closest_terminus : ", closest_terminus)
    
    return closest_terminus

def find_correct_direction2(graph: dict, current_station: int, reference: int, ligne: int) -> str:
    """
    Trouve le terminus le plus proche sur une ligne donnée pour une station actuelle.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - current_station (int): Le sommet actuel.
    - destination (int): La station de destination.
    - ligne (int): Le numéro de la ligne.
    - terminus_target (int): Le terminus cible pour la direction.

    Returns:
    - str: Le nom du terminus le plus proche.
    """
    terminus_candidates = [
        sommet for sommet, data in graph.items()
        if data['numéro_ligne'] == ligne and data['si_terminus'] is True #and data['branchement'] == graph[destination]['branchement']
    ]
    if len (terminus_candidates) == 1:
        return graph[terminus_candidates[0]]['nom_sommet']

    min_distance_to_destination = float('inf')
    closest_terminus = None
    print("current_station : ", current_station)
    print("reference : ", reference)
    print("ligne : ", ligne)
 
    print('terminus_candidates : ', terminus_candidates)
    for terminus in terminus_candidates:
        distance_from_current = count_stations_between(graph, current_station, terminus, ligne)
        distance_from_reference = count_stations_between(graph, reference, terminus, ligne)

        print("terminus : ", terminus) 
        print("distance_from_current : ", distance_from_current)
        print("distance_from_destination : ", distance_from_reference)
        print("distance_from_destination < distance_from_current : ", distance_from_reference < distance_from_current)
        print("distance_from_destination < min_distance_to_destination : ", distance_from_reference < min_distance_to_destination)

        # Vérifier si le terminus est plus proche de la destination que de la station actuelle
        if distance_from_reference > distance_from_current and distance_from_reference < min_distance_to_destination:
            min_distance_to_destination = distance_from_current
            print("min_distance_to_destination : ", min_distance_to_destination)
            closest_terminus = graph[terminus]['nom_sommet']
            print("closest_terminus : ", closest_terminus)
    
    return closest_terminus

def count_stations_between(graph: dict, start: int, end: int, ligne: int) -> int:
    """
    Compte le nombre de stations entre deux sommets sur la même ligne.
    
    Parameters:
    - graph (dict): Dictionnaire représentant le graphe.
    - start (int): Station de départ.
    - end (int): Station d'arrivée.
    - ligne (int): Le numéro de la ligne.

    Returns:
    - int: Nombre de stations entre start et end sur la ligne spécifiée.
    """
    # Effectue un parcours en largeur pour trouver le nombre de stations
    visited = set()
    queue = [(start, 0)]  # (station actuelle, nombre de stations parcourues)

    while queue:
        current_station, count = queue.pop(0)
        if current_station == end:
            return count
        
        visited.add(current_station)
        
        # Parcours des voisins
        for voisin in graph[current_station]['voisins']:
            voisin_num = voisin['num_sommet']
            if voisin_num not in visited and graph[voisin_num]['numéro_ligne'] == ligne:
                queue.append((voisin_num, count + 1))
    
    return float('inf')  # Retourne une valeur élevée si aucun chemin n'est trouvé



def generer_instructuion(graph: dict, chemin: list, distances: dict) -> list:
    print('hello wolrd')


if __name__ == "__main__":
    # Chemins vers les fichiers de données
    file_path_stations = r"asset/stations.txt"
    file_path_liaisons = r"asset/liaisons.txt"
    
    # Création des DataFrames
    stations, liaisons = create_dataframes(file_path_stations, file_path_liaisons)
    
    # Visualisation des DataFrames
    if stations is not None and liaisons is not None:
        # Afficher un aperçu des premières lignes des DataFrames
        print("Stations (aperçu) :")
        print(stations.head(), "\n")
        print("Liaisons (aperçu) :")
        print(liaisons.head(), "\n")

        # Création du graphe
        graphe = create_graphe(stations, liaisons)

        # Affichage structuré du graphe
        afficher_graphe(graphe, max_sommets=5)
    else:
        print("Erreur : Les DataFrames des stations ou des liaisons n'ont pas pu être chargés.")

    # Vérification de la connexité
    is_connected, non_visited_nodes = est_connexe(graphe)
    if is_connected:
        print("Le graphe est connexe.\n")
    else:
        print("Le graphe n'est pas connexe. Sommets non atteignables :\n", non_visited_nodes)

    # Vérification des sommets non terminus avec moins de deux voisins
    sommets_moins_deux_voisins = trouver_sommets_non_terminus_avec_moins_de_deux_voisins(graphe)
    if sommets_moins_deux_voisins:
        print(f"Sommets non terminus avec moins de deux voisins {len(sommets_moins_deux_voisins)}:\n", sommets_moins_deux_voisins)
    else:
        print("Tous les sommets non terminus ont au moins deux voisins.\n")

    # Exemple d'utilisation
    start_station = 'Carrefour Pleyel'  # ou utilisez l'ID correspondant
    end_station = 'Villejuif, P. Vaillant Couturier'  # ou l'ID correspondant

    # Trouver les IDs de station dans graphe, si nécessaire
    start_id = next((k for k, v in graphe.items() if v['nom_sommet'] == start_station), None)
    end_id = next((k for k, v in graphe.items() if v['nom_sommet'] == end_station), None)

    if start_id is not None and end_id is not None:
        instructions = bellman_ford(graphe, start_id, end_id)
        for instruction in instructions:
            print(instruction)
    else:
        print("Station de départ ou d'arrivée introuvable.")














#lignes = stations.groupby("numéro_ligne")["nom_sommet"]
#print(lignes)   

#def chemin(liaisons : pd.DataFrame, noeud1, noeud2) : 
#    file = []
#    file.append(noeud1)
#    continuer = True
#    while file or continuer :
#        voisins = liaisons[(liaisons['num_sommet1'] == examine) | (liaisons['num_sommet2'] == examine)]
#        for _, voisin in voisins.iterrows():
#            if voisin['num_sommet1'] == examine:
#                file.append(voisin['num_sommet2'])
#            else:
#                file.append(voisin['num_sommet1'])
#        examine = file.pop(0)
#        liaisons



#def connexite(graphe) : 
#    for noeud in graphe : 
#        for voisin in graphe[:noeud] : 
#            if not chemin(noeud1=noeud, noeud2=voisin) : raise "Pas connexe"
#    return "Connexe"
