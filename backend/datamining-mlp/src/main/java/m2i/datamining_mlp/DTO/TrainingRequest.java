package m2i.datamining_mlp.DTO;

// Classe représentant une requête d'entraînement pour le modèle
public class TrainingRequest {
    
    private int[] hiddenSizes; // Tableau des tailles des couches cachées
    private String activationFunction; // Nom de la fonction d'activation
    private double learningRate; // Taux d'apprentissage
    private int epochs; // Nombre d'époques (itérations d'entraînement)

    // Constructeur par défaut
    public TrainingRequest() {}

    // Constructeur avec initialisation de tous les champs
    public TrainingRequest(int[] hiddenSizes, String activationFunction, double learningRate, int epochs) {
        this.hiddenSizes = hiddenSizes;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    // Getter pour les tailles des couches cachées
    public int[] getHiddenSizes() {
        return hiddenSizes;
    }

    // Setter pour les tailles des couches cachées
    public void setHiddenSizes(int[] hiddenSizes) {
        this.hiddenSizes = hiddenSizes;
    }

    // Getter pour la fonction d'activation
    public String getActivationFunction() {
        return activationFunction;
    }

    // Setter pour la fonction d'activation
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    // Getter pour le taux d'apprentissage
    public double getLearningRate() {
        return learningRate;
    }

    // Setter pour le taux d'apprentissage
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    // Getter pour le nombre d'époques
    public int getEpochs() {
        return epochs;
    }

    // Setter pour le nombre d'époques
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
}
