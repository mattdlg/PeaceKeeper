"""
Projet Portrait robot numérique 
-------------------------------
Algorithme génétique :
    Création d'images similaires à une image 
    de départ à l'aide d'un algorithme simulant
    l'évolution d'une population par sélection 
    naturelle.
-------------------------------
Auteurs : 
    Deléglise Matthieu & Durand Julie
-------------------------------
Version : 
    4.5 (07/04/2025)
"""

#### Libraries ####
import numpy as np

class GeneticAlgorithm():
    """
    Classe GeneticAlgorithm

    Cet algorithme vise à créer un ensemble d'images similaires à une ou plusieurs 
    images cibles en utilisant les techniques d'algorithmes génétiques de croisement et de mutation.

    Les autres étapes de l'algorithme génétique (initialisation, sélection, condition d'arrêt)
    sont réalisées directement par l'utilisateur dans une interface graphique (voir new_inference.py).

    Ces images sont toutes représentées sous forme de petits tableaux, 
    qui sont la représentation des images dans l'espace latent d'un autoencodeur.

    """
    
    def __init__(self, picture1, picture2, nb_to_retrieve, mutation_rate, sigma_mutation, crossover_method = "square"):
        """
        Crée une instance de la classe GeneticAlgorithm.
        Cette instance est initialisée avec deux images cibles,
        qui seront croisées et mutées pour obtenir une nouvelle
        génération d'images semblables entre elles et aux images cibles.

        Paramètres
        ----------
        picture1 : np.array
           Tableau de taille (m,n,p) encodant une image.
        picture2 : np.array
           Tableau de taille (m,n,p) encodant une autre image.
        nb_to_retrieve : int
            Nombre de solutions (images à crées) de l'algorithme génétique que nous voulons obtenir à la fin.
        crossover_method : string
            Méthode utilisée pour échanger des coordonnées entre les deux images.
            La valeur par défault est "square". Peut aussi être "single-canal", "single-line", 
            "single-column", "uniform", "BLX-alpha" or "blending".
        mutation_rate : float ([0;1])
            Probabilité de mutation de chaque coordonnée (gène) d'un tableau (chromosome).
        sigma_mutation : float
            Ecart type de la loi normal définissant le composant aléatoire 
            ajouté aux vecteurs durant une mutation.

        Retours
        -------
        None

        """
        #### Images Initiales ####
        self.p1 = picture1
        self.p2 = picture2

        self.dimension = self.p1.shape # Dimension des vecteurs de l'espace latent = "Nombre de gènes d'un individu/chromosome"

        #### Paramètres de crossover et de mutation ####
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.sigma_mutation = sigma_mutation
        
        self.size_crossover = int(self.dimension[0]/10) # Seulement utile dans le cas où les vecteurs sont en 1D dans l'espace latent
        self.square_size = (2, 2) # Taille arbitraire d'un carré à échanger si la crossover_method est "square"

        self.solutions = self.crossover_and_mutations(nb_to_retrieve) # solutions ou GA.

    def crossover_and_mutations(self, nb_to_retrieve):
        """
        Dernière étape de chaque boucle d'un algorithme génétique :
        - reproduction entre les parents (croisement) pour donner une population d'enfants,
        - mutation sur ces enfants.

        Paramètres
        ----------
        nb_to_retrieve : int
            Nombre de solutions à créer en croisant et mutant les images initiales.

        Utilise :
        self.p1, self.p2, self.dimension
        self.crossover_method, self.square_size

        Retours
        -------
        arr_child : np.array
            Tableau de tableaux représentant la nouvelle population composée des images enfants
            créées en croisant les coordonnées des parents puis en ajoutant des mutations aléatoires.

        """
        #### Paramètres de Crossovers ####
        # ATTENTION : nb_to_retrieve doit être pair
        nb_iteration = nb_to_retrieve//2 # Nombre de reproductions à faire (2 enfants par reproduction)

        arr_crossing_points = np.zeros(nb_iteration)
        if len(self.dimension) == 1 : # cas où les vecteurs sont en 1D dans l'espace latent
            # Pour les méthodes "single", choix de nb_iteration coordonnées à échanger entre les parents pour avoir nb_iteration*2 enfants.
            if self.crossover_method == "single-coordinate":
                arr_crossing_points = np.random.randint(0, self.dimension[0], nb_iteration).astype(np.int64)

            elif self.crossover_method == "single-point" :
                arr_crossing_points = np.linspace(0.2, 0.8, nb_iteration)*self.dimension
                arr_crossing_points = np.rint(arr_crossing_points).astype(np.int64) 
        
            elif self.crossover_method == "two-points" :
                arr_lower_points = np.linspace(0.25, 0.75, nb_iteration)*self.dimension
                arr_lower_points = np.rint(arr_lower_points).astype(np.int64)
                arr_upper_points = arr_lower_points + self.size_crossover
                arr_crossing_points = np.array([(arr_lower_points[i], arr_upper_points[i]) for i in range(len(arr_lower_points))])
        
        else : # Cas où les vecteurs n'ont pas été flattened (3D)
            # Pour les méthodes "single", choix de nb_iteration canaux (resp lignes, colonnes) à échanger entre les parents pour avoir nb_iteration*2 enfants.
            if self.crossover_method == "single-canal" : 
                arr_crossing_points = np.random.choice(self.dimension[0], nb_iteration, replace=False) # échanger seulement des canaux particuliers
            elif self.crossover_method == "single-line" : 
                arr_crossing_points = np.random.choice(self.dimension[1], nb_iteration, replace=False) # échanger seulement des lignes particulières
            elif self.crossover_method == "single-column" :
                arr_crossing_points = np.random.choice(self.dimension[2], nb_iteration, replace=False) # échanger seulement des colonnes particulières
            
            # Pour la méthode "square", choix aléatoire des coordonnées du sommet haut-gauche du carré. La taille du carré est arbitrairement fixée par self.square_size
            elif self.crossover_method == "square" : # échanger seulement des carrés de coordonnées particuliers
                arr_line_index = np.random.choice(self.dimension[1]-self.square_size[0], nb_iteration, replace=False)
                arr_column_index = np.random.choice(self.dimension[2]-self.square_size[1], nb_iteration, replace=False)
                arr_crossing_points = np.array([(arr_line_index[i], arr_column_index[i]) for i in range(nb_iteration)])

        # méthode "BLX-alpha" avec différent paramètres d'exploration :
        arr_explorations = np.linspace(0.2, 0.8, nb_iteration) 

        # méthode "blending" avec un pourcentage différent des tableaux des parents à mixer :
        arr_alpha = np.linspace(0.1, 0.45, nb_iteration) # On arrête avant 0.5 car self.crossover créé deux enfants par itération -> 0.4 et 0.6 auront les mêmes résultats

        #### Reproduction ####
        list_child = []
        for k in range(nb_iteration):
            ## Crossover ##
            child1, child2 = self.crossover(self.p1, self.p2, 
                                            arr_crossing_points[k], 
                                            arr_explorations[k],
                                            arr_alpha[k])

            ## Mutations ##
            self.modified_mutation(child1)
            self.modified_mutation(child2)

            list_child.append(child1)
            list_child.append(child2)
        
        arr_child = np.asarray(list_child) # renvoie les enfants en tant qu'array d'arrays (plus simple à utiliser ensuite dans l'autoencodeur)
        return arr_child
    
    def crossover(self, parent1, parent2, crossing_point, w, alpha):
        """
        Implémentation d'un croisement entre deux tableaux parents.
        La fonction échange certaines coordonnées des vecteurs pour 
        créer deux vecteurs enfants.

        Possibilité d'utiliser des méthodes différentes pour effectuer ce croisement :
            #### Pour les tableaux aplatis uniquement ####
            - single-point : échange toutes les coordonnées après un index (inclus).
            - two-points : échange toutes les coordonnées entre une borne inférieure et une borne supérieure (incluses).

            #### Pour les tableaux 3D uniquement ####
            - single-canal : échange toutes les coordonnées dans un canal particulier des tableaux.
            - single-line : échange toutes les coordonnées dans une ligne particulière des tableaux.
            - single-column : échange toutes les coordonnées dans une colonne particulière des tableaux.
            - square : échange toutes les coordonnées dans un carré particulier (même carré dans chaque canal) des tableaux.

            #### Pour les deux types ####
            - uniform : pour chaque coordonnée : lancer une pièce pour déterminer si elle proviendra du premier ou du second parent 
            (si, à la fin de la méthode, aucune coordonnée n'a été échangée, choisir aléatoirement une coordonnée à échanger).
            - BLX-alpha : Étendre la plage de valeurs que les enfants peuvent prendre en dehors de la plage des parents,
            en utilisant un facteur alpha (ici noté w pour éviter la redondance avec le paramètre de blending). 
            Cela permet plus de diversité dans les descendants pour éviter une convergence prématurée
            de l'algorithme génétique.
            - blending : mélange des coordonnées des deux parents (combinaison linéaire) en utilisant un facteur 
            alpha définissant le pourcentage de chaque parent conservé (ex : enfant1 = 0.7parent1 + 0.3parent2).

        Paramètres
        ----------
        parent1 : np.array
            Tableau de dimension n représentant le premier parent.
        parent2 : np.array
            Tableau de dimension n représentant le second parent.
        

        Retours
        -------
        child1 : np.array
            Tableau de dimension n représentant le premier enfant.
        child2 : np.array
            Tableau de dimension n représentant le second enfant.

        """
        size = len(self.dimension)
        try :
            ### Méthode 1D ###
            if size == 1 and self.crossover_method == "single-coordinate" : # Echange seulement une unique coordonnée entre les parents 
                child1, child2 = parent1.copy(), parent2.copy()
                child1[crossing_point] = 100
                child2[crossing_point] = 100

            elif size == 1 and self.crossover_method == "single-point" : # échange toutes les coordonnées positionnées après un index aléatoire, incluant cette index.
                child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
                child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

            elif size == 1 and self.crossover_method == "two-points" : # échange toutes les coordonnées situées entre deux indexs, incluant ces indexs.
                low_crossing_point = crossing_point[0] # Limite inférieur
                high_crossing_point = crossing_point[1] # Limite supérieur (valeur défini par self.crossover_size)
                
                child1 = np.concatenate((parent1[:low_crossing_point], parent2[low_crossing_point:high_crossing_point+1], parent1[high_crossing_point+1:]))
                child2 = np.concatenate((parent2[:low_crossing_point], parent1[low_crossing_point:high_crossing_point+1], parent2[high_crossing_point+1:]))
            
            ### Méthode 3D ###
            elif size > 1 and self.crossover_method == "single-canal": 
                child1, child2 = parent1.copy(), parent2.copy()
                # Echange un canal :
                child1[crossing_point, :, :] = parent2[crossing_point, :, :]
                child2[crossing_point, :, :] = parent1[crossing_point, :, :]
            
            elif size > 1 and self.crossover_method == "single-line": 
                child1, child2 = parent1.copy(), parent2.copy()
                # Echange une ligne
                child1[:, crossing_point, :] = parent2[:, crossing_point, :]
                child2[:, crossing_point, :] = parent1[:, crossing_point, :]

            elif size > 1 and self.crossover_method == "single-column":
                # Echange une colonne
                child1, child2 = parent1.copy(), parent2.copy()
                child1[:, :, crossing_point] = parent2[:, :, crossing_point]
                child2[:, :, crossing_point] = parent1[:, :, crossing_point]

            elif size > 1 and self.crossover_method == "square": 
                child1, child2 = parent1.copy(), parent2.copy()
                lower_line_index, upper_line_index = crossing_point[0], crossing_point[0]+self.square_size[0]
                lower_column_index, upper_column_index = crossing_point[1], crossing_point[1]+self.square_size[1]
                # Echange un carré de valeur (le même dans chaque canal)
                child1[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index] = parent2[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index]
                child2[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index] = parent1[:, lower_line_index:upper_line_index, lower_column_index:upper_column_index]

            ### Méthode commune 1D-3D ###
            elif self.crossover_method == "uniform" : # Choisi aléatoire de quel parent vient une coordonnée
                child1, child2 = np.zeros(self.dimension, dtype=np.float32), np.zeros(self.dimension, dtype=np.float32)
                mask = np.random.randint(0, 2, size=self.dimension, dtype=bool) # lance une pièce pour chaque coordonnée : pile = coordonnée de p1, face = coordonnée de p2
                child1, child2 = np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)

                if np.array_equal(child1, parent1) : # Lors d'un évènement de crossover, au moins une coordonnée doit être modifié
                    random_index = np.random.randint(self.dimension) # choix aléatoire d'une coordonnée pour s'assurer que les enfants sont différents des parents.
                    child1[random_index] = parent2[random_index]
                    child2[random_index] = parent1[random_index]
            
            elif self.crossover_method == "BLX-alpha": # Blend Alpha crossover
                # Calcul des bornes inférieures et supérieures par élargissement de la plage de valeur des parents.
                lower = np.minimum(parent1, parent2) - w * np.abs(parent1 - parent2) 
                upper = np.maximum(parent1, parent2) + w * np.abs(parent1 - parent2) 
                # Créer aléatoirement les coordonnées des enfants entre les limites :
                child1 = np.random.uniform(lower, upper)
                child2 = np.random.uniform(lower, upper)

            elif self.crossover_method == "blending": # Combinaison linéaire des coordonnées des parents
                # child1 garde alpha*100% de parent1 et (1-alpha)*100% de parent2, et inversement pour child2 :
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2

            else : 
                raise ValueError("Méthode de Crossover non reconnue. Merci de choisir une méthode parmi les suivantes : " \
                "single-coordinate, single-point, two-points (vecteurs 1D); " \
                "single-canal, single-line, single-column, square (vecteurs 3D); " \
                "uniform, BLX-alpha, blending (1D ou 3D).")
                # child1, child2 = parent1, parent2 # keep children equal to parents if a wrong method is called

            return child1, child2
        
        except ValueError as e:
            print("Erreur de méthode de croissement : ", e)

    
    def mutation(self, chr):
        """
        Implémentation d'un événement de mutation sur un chromosome/individu (tableau).
        Chaque gène (coordonnée) du chromosome a une probabilité de muter en fonction
        du taux de mutation donné en paramètre de la classe.
        Dans cette version de la méthode de mutation, différents vecteurs de mutation 
        sont ajoutés à chaque canal (première dimension). 
        Les vecteurs 1D utilisent forcément cette méthode (une seule dimension).

        Les mutations sont tirées d'une distribution normale centrée sur 0 
        et d'écart type donnée en paramètre de la classe.
        Modifier l'écart type sigma permet d'avoir des mutations plus grandes ou plus petites.

        Paramètres
        ----------
        chr : np.array
            Tableau représentant une image encodée.
        
        Utilise :
        self.dimension, self.mutation_rate et self.sigma_mutation

        Retours
        -------
        None
            chr est modifié directement par la fonction et n'a donc pas besoin d'être retourné.

        """
        mask = np.random.random_sample(self.dimension) < self.mutation_rate # Chaque coordonnée a une probabilité de muter
        mutations = np.random.normal(0, self.sigma_mutation, self.dimension) # Les mutations sont tirées d'une distribution normale
        chr[mask] += mutations[mask] # Addition de la composante aléatoire au vecteur initial lorsque ses coordonnées ont muté.

    def modified_mutation(self, chr):
        """
        Implémentation d'un événement de mutation sur un chromosome/individu (tableau).
        Chaque gène (coordonnée) du chromosome a une probabilité de muter en fonction
        du taux de mutation donné en paramètre de la classe.
        Dans cette version de la méthode de mutation, le même vecteur de mutation est ajouté
        à chaque canal de l'image (première dimension). 
        Les vecteurs 1D n'ayant qu'une seule dimension, ils ne sont pas affectés par cette méthode 
        et appel forcément la méthode self.mutation d'origine.

        Les mutations sont tirées d'une distribution normale centrée sur 0S
        et d'écart type donnée en paramètre de la classe.
        Modifier l'écart type sigma permet d'avoir des mutations plus grandes ou plus petites.

        Paramètres
        ----------
        chr : np.array
            Tableau représentant une image encodée.
        
        Utilise :
        self.dimension, self.mutation_rate et self.sigma_mutation

        Retours
        -------
        None
            chr est modifié directement par la fonction et n'a donc pas besoin d'être retourné.

        """
        if len(self.dimension) > 1 : # Dans le cas où le tableau est 3D, applique le même vecteur de mutation sur tous les cas
            mask = np.random.random_sample(self.dimension[1:]) < self.mutation_rate
            mutations = np.random.normal(0, self.sigma_mutation, self.dimension[1:]) 
            for i in range(self.dimension[0]):
                chr[i][mask] += mutations[mask] 
        
        else : # Si le tableau est en 1D, pas de notion de canal : applique la méthode initiale
            self.mutation(chr)


def test_mutation(picture1, picture2, p_mut=0.5, s_mut=1, shape = "unflat"):
    """
    Fonction de test des mutations de l'algorithme génétique.
    Cette fonction crée une instance de la classe GeneticAlgorithm, 
    puis un vecteur aléatoire de même taille que les images d'entrées du GA.
    Elle applique ensuite la méthode de mutation sur ce vecteur aléatoire.

    Paramètres
    ----------
    picture1 : np.array
        Tableau de dimension n représentant le premier parent.
    picture2 : np.array
        Tableau de dimension n représentant le second parent.
    p_mut : float
        Probabilités de mutation de chaque coordonnée d'un tableau.
    s_mut : float
        écart type de la loi normal définissant les vecteurs de mutation.
    shape : str
        Indique si les tableaux d'entrées sont aplatis ou non.
        La valeur par défault est "unflat". Peut aussi être "flat". 
        Pour tout autre valeur, la fonction lève une erreur.

    Retours
    -------
    None 
        Affiche les tableaux avant et après mutation.

    Exemples
    --------
    >>> np.random.seed(42)
    >>> a = np.random.rand(2,2,2) * 10
    >>> b = np.random.rand(2,2,2) * 10
    >>> test_mutation(a,b,0.5,1,"flat")
    Before mutation: 
    [8.07440155 8.960913   3.18003475 1.10051925 2.27935163 4.27107789
    8.18014766 8.60730583]
    After mutation:
    [ 9.96058745  8.960913    3.43758514  1.02607333  0.36058041  4.24456401
    8.18014766 11.07054795]

    >>> test_mutation(a,b,0.5,1,"unflat")
    Before mutation: 
    [[[0.74550644 9.86886937]
      [7.72244769 1.98715682]]

     [[0.05522117 8.15461428]
      [7.06857344 7.29007168]]]
    After mutation:
    [[[ 0.74550644 10.84441449]
      [ 7.24327346  1.80149784]]

     [[ 0.05522117  9.13015941]
      [ 6.5893992   7.1044127 ]]]

    """
    #### Vérification des paramètres ####
    if picture1.shape != picture2.shape:
        raise ValueError("Les deux tableaux d'entrées doivent avoir la même taille.")
    if shape not in ["unflat", "flat"]:
        raise ValueError("La valeur de shape doit être 'unflat' ou 'flat'.")
    if p_mut < 0 or p_mut > 1:
        raise ValueError("La probabilité de mutation doit être comprise entre 0 et 1.")
    if s_mut < 0:
        raise ValueError("L'écart type de la mutation doit être positif.")
    
    if shape == "flat":
        picture1 = picture1.flatten(order="C")
        picture2 = picture2.flatten(order="C")

    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=6, crossover_method="blending", 
                          mutation_rate=p_mut, sigma_mutation=s_mut)
    
    chr = np.random.rand(*picture1.shape) * 10 # Création d'un tableau aléatoire de même taille que les images d'entrées du GA
    
    print("Before mutation: ")
    print(chr) 
    ga.modified_mutation(chr)
    print("After mutation: ")
    print(chr)
    print("\n")


def test_crossover(picture1, picture2, point_of_crossover, w, alpha, method):
    """
    Fonction de test des crossover de l'algorithme génétique.
    Cette fonction crée une instance de la classe GeneticAlgorithm, 
    Elle applique ensuite une méthode de crossover sur les deux vecteurs d'origines.

    Paramètres
    ----------
    picture1 : np.array
        Tableau de dimension n représentant le premier parent.
    picture2 : np.array
        Tableau de dimension n représentant le premier parent.
    point_of_crossover : int ou tuple
        index de croisement pour les méthodes "single-point", "single-coordinate",
        "single-canal", "single-line", "single-column" (int) ou 
        coordonnées de croisement pour les méthodes "two-points", "square" (tuple).
    w : float
        Paramètre d'exploration pour la méthode "BLX-alpha".
    alpha : float
        Paramètre de mélange pour la méthode "blending".
    method : str
        Méthode de crossover à utiliser pour croiser les parents.

    Retours 
    -------
    None
        Affiche les coordonnées des parents et des enfants après crossover.

    Exemples
    --------
    >>> np.random.seed(42)
    >>> a = np.random.rand(8) * 10
    >>> b = np.random.rand(8) * 10
    >>> test_crossover(a, b, 1, 0.5, 0.3, "single-point")
    ---- Parents ----
    [3.74540119 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    [6.01115012 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    ---- Children ----
    [3.74540119 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    [6.01115012 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]

    >>> test_crossover(a, b, 1, 0.5, 0.3, "blending")
    ---- Parents ----
    [3.74540119 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    [6.01115012 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    ---- Children ----
    [5.33142544 7.80865096 2.34007329 8.58534442 6.29515441 1.95435734
    1.44702561 3.88236001]
    [4.42512587 8.77921788 5.18571108 7.10033895 3.58945841 1.72897897
    0.95206019 6.61344655]

    """
    #### Vérification des paramètres ####
    if picture1.shape != picture2.shape:
        raise ValueError("Les deux tableaux d'entrées doivent avoir la même taille.")
    if method not in ["single-coordinate", "single-point", "two-points", 
                                 "single-canal", "single-line", "single-column", 
                                 "square", "uniform", "BLX-alpha", "blending"]:
        raise ValueError("La méthode de croisement est inconnue")
    
    if method in ["single-coordinate", "single-point", "single-canal", 
                  "single-line", "single-column"] and not isinstance(point_of_crossover, int):
        raise TypeError("La méthode de croisement demandée nécessite un index entier.")
    if method in ["two-points", "square"] and not isinstance(point_of_crossover, tuple):
        raise TypeError("La méthode de croisement demandée nécessite un tuple de coordonnées.")
    if method == "BLX-alpha" and (w < 0 or w > 1):
        raise ValueError("Le paramètre d'exploration w doit être compris entre 0 et 1.")
    if method == "blending" and (alpha < 0 or alpha > 1):
        raise ValueError("Le paramètre de mélange alpha doit être compris entre 0 et 1.")

    ga = GeneticAlgorithm(picture1, picture2, nb_to_retrieve=2, crossover_method=method, 
                          mutation_rate=0.5, sigma_mutation=1)
    
    print("---- Parents ----")
    print(picture1)
    print(picture2)
    child1, child2 = ga.crossover(picture1, picture2, point_of_crossover, w, alpha)
    print("---- Children ----")
    print(child1)
    print(child2)

def run_ga(targets, nb_solutions, crossover_method, mutation_rate, sigma_mutation):
    """
    Fonction permettant de lancer l'algorithme génétique sur deux images cibles.
    Cette fonction crée une instance de la classe GeneticAlgorithm,
    puis récupère les solutions (enfants) de l'algorithme génétique.
    Ces images sont toutes représentées sous forme de tableaux.

    Paramètres
    ----------
    targets : list
        Liste des tableaux des parents à donner à l'algorithme génétique.
    nb_solutions : int
        Nombre d'enfants à créer en utilisant le croisement et la mutation dans l'algorithme génétique.
    crossover_method : string
        Méthode de croisement à utiliser pour mélanger les coordonnées des parents et obtenir de nouveaux enfants.
    mutation_rate : float
        Probabilité de mutation de chaque coordonnée d'un tableau.
    sigma_mutation : float
        Écart type de la distribution normale utilisée pour créer des vecteurs de mutation aléatoires.

    Retours
    -------
    solutions : np.array
        Tableau de tableaux représentant les solutions de l'algorithme génétique.

    Exemples 
    --------
    >>> np.random.seed(42)
    >>> a = np.random.rand(2,2,2) * 10
    >>> b = np.random.rand(2,2,2) * 10
    >>> list_target = [a.flatten(order="C"), b.flatten(order="C")]
    >>> solutions = run_ga(list_target, nb_solutions=6, crossover_method="blending", mutation_rate=0.5, sigma_mutation=1)
    >>> print_solutions_coordinates(list_target, solutions)
    ---- Parents ----
    [3.74540119 9.50714306 7.31993942 5.98658484 1.5601864  1.5599452
    0.58083612 8.66176146]
    [6.01115012 7.08072578 0.20584494 9.69909852 8.32442641 2.12339111
    1.81824967 1.8340451 ]
    ---- Children ----
    [7.25022399 7.32336751 0.9847826  7.90309897 7.64800241 2.17796911
    0.54351474 2.89251475]
    [ 3.95847886  9.26450134  6.60852997  6.35783621  2.445474   -0.34338033
    0.70457748  8.17585106]
    [5.38806916 7.74799053 2.16222092 8.67815726 6.46426041 1.96844348
    1.82157924 1.94862694]
    [5.39948167 8.83987831 4.52434591 7.0075261  3.42035241 2.69043795
    0.44195061 6.78413946]
    [5.3529587  9.71065012 3.40718746 8.02846737 5.28051841 2.69174295
    1.34846064 4.90651746]
    [3.9564946  8.41525529 4.1185969  7.98596711 4.0743342  1.81349586
    1.23474977 6.55793409]

    """
    #### Vérification des paramètres ####
    if len(targets) != 2:
        raise ValueError("La liste des cibles doit contenir exactement deux tableaux.")
    if targets[0].shape != targets[1].shape:
        raise ValueError("Les deux tableaux d'entrées doivent avoir la même taille.")
    if nb_solutions % 2 != 0:
        raise ValueError("Le nombre de solutions doit être pair.")
    if crossover_method not in ["single-coordinate", "single-point", "two-points", 
                                 "single-canal", "single-line", "single-column", 
                                 "square", "uniform", "BLX-alpha", "blending"]:
        raise ValueError("La méthode de croisement est inconnue")
    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError("Le taux de mutation doit être compris entre 0 et 1.")
    if sigma_mutation < 0:
        raise ValueError("L'écart type de la mutation doit être positif.")
    
    ga = GeneticAlgorithm(targets[0], targets[1], nb_to_retrieve=nb_solutions, crossover_method=crossover_method, 
                          mutation_rate=mutation_rate, sigma_mutation=sigma_mutation)
    
    solutions = ga.solutions
    # solutions = solutions.reshape((-1, *dimensions), order="C") 
    return solutions


def print_solutions_coordinates(list_target, solutions):
    """
    Fonction d'affichage des coordonnées des parents et des enfants.
    Cette fonction affiche les coordonnées des parents et des enfants après croisement et mutation.

    Paramètres
    ----------
    list_target : list
        Liste des vecteurs des images encodées données en entrée de l'Algorithme Génétique
    solutions : np.array
        Tableau des solutions du GA.

    Retours 
    -------
    None 
        Affiche les coordonnées des parents et des enfants.

    """
    print("---- Parents ----")
    print(list_target[0])
    print(list_target[1])
    print("---- Children ----")
    for s in solutions :
        print(s)

if __name__ == "__main__" :
    print(__name__)
    
    np.random.seed(42)
    target = np.random.rand(2,2,2) * 10
    target2 = np.random.rand(2,2,2) * 10
    try:
        str_user = input("Que voulez vous faire ? (test mutation, test crossover, run GA) : ").lower().strip()
        if str_user == "test mutation":
            try :
                test_mutation(target, target2, 0.5, 1, "flat")
            except Exception as e:
                print(f"Erreur : {e}")
        elif str_user == "test crossover":
            try :
                test_crossover(target.flatten(order="C"), target2.flatten(order="C"), 1, 0.5, 0.3, "single-point")
            except Exception as e:
                print(f"Erreur : {e}")
        elif str_user == "run ga":
            list_target = [target.flatten(order="C"), target2.flatten(order="C")]
            try :
                solutions = run_ga(list_target, nb_solutions=6, crossover_method="blending", 
                                mutation_rate=0.5, sigma_mutation=1)
                print_solutions_coordinates(list_target, solutions)
            except Exception as e:
                print(f"Erreur : {e}")

        else:
            raise ValueError("Unknown method, please choose 'test mutation', 'test crossover' or 'run GA'.")
    except ValueError as e:
        print(f"Erreur : {e}")
    




    
