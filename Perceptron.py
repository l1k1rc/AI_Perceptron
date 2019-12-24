"""
Ce code permet la construction d'un réseau de neurone permettant la reconnaissance d'une lettre A ou C représentée
par une matrice 5*4 (l*c) de valeurs binaires. Les sorties des neurones seront représentées par des valeurs
booléennes (-1 ou 1).
Le training set est basé sur des matrices de A et de C.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()

#################################################################################################################################################

letterA = np.array([
    0, 1, 1, 0,
    1, 0, 0, 1,
    1, 1, 1, 1,
    1, 0, 0, 1,
    1, 0, 0, 1])
letter2 = np.array([
    1, 1, 1, 1,
    1, 0, 0, 1,
    1, 1, 1, 1,
    1, 0, 0, 1,
    1, 0, 0, 1
])
letterC = np.array([
    1, 1, 1, 1,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 1, 1, 1,
])

basic_weight = np.random.randint(-1, 1, 20)


class Perceptron__:
    """Classe neurone caractérisée par :
    - ses entrées
    - sa sortie
    - ses poids
    - ses poids optimaux (quand entrainé)"""

    def __init__(self, n_inputs, n_weigths, output):
        self.inputs = n_inputs
        self.w = n_weigths
        self.output = output
        self.optimal_weight = []

    def function_weight(self, w, otp_wanted, actual_otp, X):
        """
        1- Pour chaque poids w0,w1,w2,w3 et chaque x0,x1,x2,x3
        2- on applique la loi de Widrow-Hoff nouveau_poids[i]=poids[i]+epsilon*((sortie_attendues-sortie_actuelle)*entrée[i]
        3- on recalcule y avec ces nouveaux poids
        4- si le y correspond à la sortie attendues, on le stocke dans une liste de poids optimaux sinon on réitère la
           méthode jusqu'à ce que y correspond
        :param w: la matrice de poids initiale
        :param otp_wanted: la sortie voulue
        :param actual_otp: la sortie actuelle
        :param X: les entrée x0,x1,x2,x3
        :return: les poids permettant d'avoir la sortie voulue sinon récursivité de la méthode
        """
        print(str(w))
        new_w = []
        epsilon = 0.1
        useless_index = 0
        # La boucle est répétée pour le même neurone et pour chaque poids de chaque entrée du neurone
        for wght, e in zip(w, X):  # w = tableau de poids et X = tableau d'entrée pour un neurone
            print('FORMULE ERREUR w n°' + str(useless_index) + ' : ' + str(wght), str(otp_wanted), str(actual_otp),
                  str(e))
            new_w.append(wght + epsilon * ((otp_wanted - actual_otp) * e))
            useless_index += 1
        print(new_w)  # affiche le nouveau tableau de poids
        y = self.activity_heaviside(X, new_w)
        print('_______________________________________________NEW FINAL IS OK ? : ' + str(y))
        if y != otp_wanted:
            self.function_weight(new_w, otp_wanted, actual_otp, X)
        else:
            self.w = basic_weight  # on reset les poids pour le neurone suivant
            self.optimal_weight = new_w  # on ajoute les poids trouvés à une liste de stockage pour ce neurone

    def activity_heaviside(self, inp, wght, outputs=0):
        """
        Fait la somme des produit poids[i]*x[i]
        :param inp: tableau d'entrées
        :param wght: tableau de poids
        :param outputs: résultat de la somme initalisé à 0
        :return: la valeur y 1:-1
        """
        for inputs, weight in zip(inp, wght):
            outputs += inputs * weight
        return 1 if outputs >= 0 else -1

    def training(self):
        """
        Entrainement du neurone avec fct activité et fct erreur widrow-hoff.
        :return:
        """
        print('________________________________________________________________________')
        print('Valeur des inputs envoyées : ' + str(self.inputs))
        y = self.activity_heaviside(self.inputs, self.w)
        print(
            '***********************Valeur obtenue pour ce neurone ************************  ' + str(
                y) + ' et poids :' + str(self.w))
        if y != self.output:
            self.function_weight(self.w, self.output, y, self.inputs)
        else:
            self.optimal_weight.append(self.w)
            self.w = basic_weight

    def predict(self, weight):
        print(self.activity_heaviside(self.inputs, weight))

    def toString(self):
        print("************************************************************************************************\n"
              "Perceptron class :: TRAINING_SET :\n -- valeurs \n" + str(self.inputs) +
              "\n -- sorties attendues \n" + str(self.output) +
              "\n -- poids de départ \n" + str(self.w) +
              "\n -- poids trouvés dans l'ordre donné des neurones \n" + str(np.array(self.optimal_weight)) +
              "\n************************************************************************************************")

    def displayChart(self):
        objects = (
            'w0', 'w1', 'w2', "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15", "w16",
            "w17",
            "w18", "w19")
        for i in range(len(self.w)):
            performance = self.optimal_weight[i] - self.w
        y_pos = np.arange(len(objects))

        plt.barh(y_pos, performance, align='center', alpha=0.5)
        plt.yticks(y_pos, objects)
        plt.xlabel('Variation du poids Ajout/retrait final')
        plt.ylabel('Poids n')
        plt.title('Évolution de l\'erreur du réseau')
        plt.show()


"""  Perceptron__( inputs , weight , outputs_wanted)  """

perceptron_A = Perceptron__(letterA, basic_weight, 1)
perceptron_C = Perceptron__(letterC, basic_weight, -1)
perceptron_A.training()
#perceptron_A.w=perceptron_A.optimal_weight # pour réïtérer avec poids optimaux précedents ... etc
#perceptron_A.inputs=letter2
#perceptron_A.training()
perceptron_C.training()
perceptron_A.toString()
perceptron_C.toString()
print("######################################")
perceptron_A.predict(perceptron_A.optimal_weight)
perceptron_A.predict(perceptron_C.optimal_weight[0])
perceptron_A.displayChart()
