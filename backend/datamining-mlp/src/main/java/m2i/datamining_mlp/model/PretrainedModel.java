package m2i.datamining_mlp.model;

// Importation de la classe TrainingResponse utilisée pour stocker les métriques d'entraînement
import m2i.datamining_mlp.DTO.TrainingResponse;
// Importation de l'annotation @Id pour identifier l'objet dans MongoDB
import org.springframework.data.annotation.Id;
// Annotation pour indiquer que cette classe est un document MongoDB
import org.springframework.data.mongodb.core.mapping.Document;

import java.io.Serializable;
import java.util.Arrays;

// Indique que cette classe représente un document de la collection "pretrained_models"
@Document(collection = "pretrained_models")
public class PretrainedModel implements Serializable {
    // Identifiant unique fixe pour le modèle préentraîné
    @Id
    private String id = "pretrained_model"; // ID fixe pour un seul modèle préentraîné
    private double[][][] weights; // Poids du réseau de neurones
    private double[][] biases; // Biais du réseau de neurones
    private int inputSize; // Taille de l'entrée
    private int[] hiddenSizes; // Tailles des couches cachées
    private double learningRate; // Taux d'apprentissage
    private String activationFunction; // Fonction d'activation utilisée
    private TrainingResponse.TrainingMetrics metrics; // Métriques d'entraînement associées

    // Constructeur vide par défaut
    public PretrainedModel() {}

    // Constructeur prenant un classificateur et ses métriques d'entraînement
    public PretrainedModel(Classifier classifier, TrainingResponse.TrainingMetrics metrics) {
        this.weights = classifier.getWeights(); // Récupération des poids depuis le classificateur
        this.biases = classifier.getBiases(); // Récupération des biais depuis le classificateur
        this.inputSize = classifier.getInputSize(); // Taille d'entrée du classificateur
        this.hiddenSizes = classifier.getHiddenSizes(); // Tailles des couches cachées
        this.learningRate = classifier.getLearningRate(); // Taux d'apprentissage
        this.activationFunction = classifier.getActivationFunction().toString(); // Fonction d'activation convertie en chaîne
        this.metrics = metrics; // Enregistrement des métriques
    }

    // Getters et setters pour tous les champs
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public double[][][] getWeights() { return weights; }
    public void setWeights(double[][][] weights) { this.weights = weights; }

    public double[][] getBiases() { return biases; }
    public void setBiases(double[][] biases) { this.biases = biases; }

    public int getInputSize() { return inputSize; }
    public void setInputSize(int inputSize) { this.inputSize = inputSize; }

    public int[] getHiddenSizes() { return hiddenSizes; }
    public void setHiddenSizes(int[] hiddenSizes) { this.hiddenSizes = hiddenSizes; }

    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }

    public String getActivationFunction() { return activationFunction; }
    public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }

    public TrainingResponse.TrainingMetrics getMetrics() { return metrics; }
    public void setMetrics(TrainingResponse.TrainingMetrics metrics) { this.metrics = metrics; }

    // Méthode pour convertir cet objet en une instance de Classifier
    public Classifier toClassifier() {
        Classifier classifier = new Classifier(inputSize, hiddenSizes, learningRate,
                Classifier.ActivationFunction.valueOf(activationFunction)); // Création d'un nouveau classificateur
        classifier.setWeights(weights); // Attribution des poids
        classifier.setBiases(biases); // Attribution des biais
        return classifier; // Retour du classificateur reconstruit
    }
}
