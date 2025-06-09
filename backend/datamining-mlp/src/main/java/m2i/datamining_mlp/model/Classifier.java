package m2i.datamining_mlp.model;

// Annotations Lombok pour générer automatiquement les getters et setters
import lombok.Getter;
import lombok.Setter;

import java.util.Random;

@Setter
@Getter
public class Classifier {
    // Architecture du réseau
    private int inputSize; // Taille de la couche d'entrée
    private int[] hiddenSizes;  // Tableau contenant la taille de chaque couche cachée
    private int outputSize; // Taille de la couche de sortie (1 pour classification binaire)
    private int numHiddenLayers; // Nombre de couches cachées

    // Enumération des fonctions d’activation disponibles
    public enum ActivationFunction {
        SIGMOID, TANH, RELU, LEAKY_RELU
    }

    private ActivationFunction activationFunction; // Fonction d’activation utilisée

    // Poids et biais - représentés par des tableaux pour plusieurs couches
    private double[][][] weights;  // [couche][depuis][vers]
    private double[][] biases;     // [couche][neurone]

    // Taux d’apprentissage
    private double learningRate;

    // Générateur aléatoire
    private Random random;

    // Constructeur du classificateur
    public Classifier(int inputSize, int[] hiddenSizes, double learningRate, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes.clone();
        this.numHiddenLayers = hiddenSizes.length;
        this.outputSize = 1; // Classification binaire (spam ou non)
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.random = new Random(42); // Graine fixe

        initializeWeights(); // Initialisation des poids et biais
    }

    /**
     * Initialise les poids et les biais avec de petites valeurs aléatoires
     */
    private void initializeWeights() {
        int totalLayers = numHiddenLayers + 1; // Couches cachées + couche de sortie
        weights = new double[totalLayers][][];
        biases = new double[totalLayers][];

        // Boucle sur chaque couche pour initialiser poids et biais
        for (int layer = 0; layer < totalLayers; layer++) {
            int inputSizeForLayer, outputSizeForLayer;

            if (layer == 0) {
                // Première couche cachée
                inputSizeForLayer = inputSize;
                outputSizeForLayer = hiddenSizes[0];
            } else if (layer < numHiddenLayers) {
                // Couches cachées suivantes
                inputSizeForLayer = hiddenSizes[layer - 1];
                outputSizeForLayer = hiddenSizes[layer];
            } else {
                // Couche de sortie
                inputSizeForLayer = hiddenSizes[numHiddenLayers - 1];
                outputSizeForLayer = outputSize;
            }

            // Initialisation des poids
            weights[layer] = new double[inputSizeForLayer][outputSizeForLayer];
            for (int i = 0; i < inputSizeForLayer; i++) {
                for (int j = 0; j < outputSizeForLayer; j++) {
                    weights[layer][i][j] = random.nextGaussian() * 0.1;
                }
            }

            // Initialisation des biais
            biases[layer] = new double[outputSizeForLayer];
            for (int i = 0; i < outputSizeForLayer; i++) {
                biases[layer][i] = random.nextGaussian() * 0.1;
            }
        }
    }

    /**
     * Applique la fonction d’activation choisie
     */
    private double activate(double x, ActivationFunction function) {
        switch (function) {
            case SIGMOID:
                return 1.0 / (1.0 + Math.exp(-x));
            case TANH:
                return Math.tanh(x);
            case RELU:
                return Math.max(0, x);
            case LEAKY_RELU:
                return x > 0 ? x : 0.01 * x;
            default:
                return 1.0 / (1.0 + Math.exp(-x)); // Par défaut : sigmoid
        }
    }

    /**
     * Calcule la dérivée de la fonction d’activation
     */
    private double activationDerivative(double x, ActivationFunction function) {
        switch (function) {
            case SIGMOID:
                double sigmoid = activate(x, ActivationFunction.SIGMOID);
                return sigmoid * (1 - sigmoid);
            case TANH:
                double tanh = activate(x, ActivationFunction.TANH);
                return 1 - tanh * tanh;
            case RELU:
                return x > 0 ? 1 : 0;
            case LEAKY_RELU:
                return x > 0 ? 1 : 0.01;
            default:
                double sigmoidDefault = activate(x, ActivationFunction.SIGMOID);
                return sigmoidDefault * (1 - sigmoidDefault);
        }
    }

    /**
     * Propagation avant
     * @param input Entrée (fréquence des mots dans l'email)
     * @return Probabilité prédite (entre 0 et 1, >0.5 = spam)
     */
    public double predict(double[] input) {
        double[] currentInput = input.clone();

        // Propagation à travers toutes les couches
        for (int layer = 0; layer < numHiddenLayers + 1; layer++) {
            double[] nextLayer = new double[weights[layer][0].length];

            for (int j = 0; j < nextLayer.length; j++) {
                double sum = biases[layer][j];
                for (int i = 0; i < currentInput.length; i++) {
                    sum += currentInput[i] * weights[layer][i][j];
                }

                // Application de la fonction d’activation
                if (layer < numHiddenLayers) {
                    nextLayer[j] = activate(sum, activationFunction);
                } else {
                    nextLayer[j] = activate(sum, ActivationFunction.SIGMOID); // Sigmoid pour la sortie
                }
            }

            currentInput = nextLayer;
        }

        return currentInput[0]; // Sortie unique
    }

    /**
     * Entraîne le réseau sur un seul exemple
     * @param input Caractéristiques de l’email
     * @param target Étiquette vraie (0=non spam, 1=spam)
     */
    public void trainSample(double[] input, int target) {
        // PROPAGATION AVANT - on stocke les valeurs intermédiaires
        double[][] layerOutputs = new double[numHiddenLayers + 2][]; // +2 pour l’entrée et la sortie
        double[][] layerInputs = new double[numHiddenLayers + 1][]; // +1 pour les couches cachées + sortie

        layerOutputs[0] = input.clone(); // Couche d’entrée

        // Propagation vers l’avant
        for (int layer = 0; layer < numHiddenLayers + 1; layer++) {
            int inputSize = layerOutputs[layer].length;
            int outputSize = weights[layer][0].length;

            layerInputs[layer] = new double[outputSize];
            layerOutputs[layer + 1] = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                double sum = biases[layer][j];
                for (int i = 0; i < inputSize; i++) {
                    sum += layerOutputs[layer][i] * weights[layer][i][j];
                }

                layerInputs[layer][j] = sum;

                // Activation
                if (layer < numHiddenLayers) {
                    layerOutputs[layer + 1][j] = activate(sum, activationFunction);
                } else {
                    layerOutputs[layer + 1][j] = activate(sum, ActivationFunction.SIGMOID);
                }
            }
        }

        // PROPAGATION ARRIÈRE - calcul des gradients
        double[][] deltas = new double[numHiddenLayers + 1][];

        // Calcul de l’erreur pour la couche de sortie
        int outputLayerIndex = numHiddenLayers;
        deltas[outputLayerIndex] = new double[outputSize];
        double output = layerOutputs[outputLayerIndex + 1][0];
        double outputError = target - output;
        deltas[outputLayerIndex][0] = outputError * output * (1 - output); // Dérivée de sigmoid

        // Backpropagation pour les couches cachées
        for (int layer = numHiddenLayers - 1; layer >= 0; layer--) {
            deltas[layer] = new double[hiddenSizes[layer]];

            for (int i = 0; i < hiddenSizes[layer]; i++) {
                double error = 0.0;

                // Somme des erreurs provenant de la couche suivante
                for (int j = 0; j < deltas[layer + 1].length; j++) {
                    error += deltas[layer + 1][j] * weights[layer + 1][i][j];
                }

                // Dérivée de la fonction d’activation
                deltas[layer][i] = error * activationDerivative(layerInputs[layer][i], activationFunction);
            }
        }

        // MISE À JOUR DES POIDS ET BIAIS
        for (int layer = 0; layer < numHiddenLayers + 1; layer++) {
            // Mise à jour des poids
            for (int i = 0; i < weights[layer].length; i++) {
                for (int j = 0; j < weights[layer][i].length; j++) {
                    weights[layer][i][j] += learningRate * deltas[layer][j] * layerOutputs[layer][i];
                }
            }

            // Mise à jour des biais
            for (int j = 0; j < biases[layer].length; j++) {
                biases[layer][j] += learningRate * deltas[layer][j];
            }
        }
    }

    // Getters pour les paramètres du réseau
    public int getInputSize() { return inputSize; }
    public int[] getHiddenSizes() { return hiddenSizes.clone(); }
    public int getNumHiddenLayers() { return numHiddenLayers; }
    public ActivationFunction getActivationFunction() { return activationFunction; }
    public double getLearningRate() { return learningRate; }
}
