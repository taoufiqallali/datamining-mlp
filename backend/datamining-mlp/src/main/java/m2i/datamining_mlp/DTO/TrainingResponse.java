package m2i.datamining_mlp.DTO;

import java.util.List;

// Classe représentant la réponse après l'entraînement du modèle
public class TrainingResponse {
    private String status; // Statut de la réponse
    private String message; // Message associé
    private TrainingMetrics metrics; // Métriques d'entraînement
    private List<EpochLoss> epochLosses; // Liste des pertes par époque

    // Classe interne représentant les métriques d'entraînement
    public static class TrainingMetrics {
        private int totalEmails; // Nombre total d'e-mails
        private int spamEmails; // Nombre d'e-mails spam
        private int nonSpamEmails; // Nombre d'e-mails non-spam
        private int featureDimensions; // Nombre de dimensions du vecteur de caractéristiques
        private int trainSize; // Taille du jeu d'entraînement
        private int testSize; // Taille du jeu de test
        private double accuracy; // Précision du modèle
        private double spamDetectionRate; // Taux de détection des spams
        private double nonSpamDetectionRate; // Taux de détection des non-spams
        private int[] hiddenLayerSizes; // Tailles des couches cachées
        private int numHiddenLayers; // Nombre de couches cachées
        private String activationFunction; // Fonction d'activation utilisée

        public TrainingMetrics() {}

        // Accesseurs (getters/setters) 
        public int getTotalEmails() { return totalEmails; }
        public void setTotalEmails(int totalEmails) { this.totalEmails = totalEmails; }

        public int getSpamEmails() { return spamEmails; }
        public void setSpamEmails(int spamEmails) { this.spamEmails = spamEmails; }

        public int getNonSpamEmails() { return nonSpamEmails; }
        public void setNonSpamEmails(int nonSpamEmails) { this.nonSpamEmails = nonSpamEmails; }

        public int getFeatureDimensions() { return featureDimensions; }
        public void setFeatureDimensions(int featureDimensions) { this.featureDimensions = featureDimensions; }

        public int getTrainSize() { return trainSize; }
        public void setTrainSize(int trainSize) { this.trainSize = trainSize; }

        public int getTestSize() { return testSize; }
        public void setTestSize(int testSize) { this.testSize = testSize; }

        public double getAccuracy() { return accuracy; }
        public void setAccuracy(double accuracy) { this.accuracy = accuracy; }

        public double getSpamDetectionRate() { return spamDetectionRate; }
        public void setSpamDetectionRate(double spamDetectionRate) { this.spamDetectionRate = spamDetectionRate; }

        public double getNonSpamDetectionRate() { return nonSpamDetectionRate; }
        public void setNonSpamDetectionRate(double nonSpamDetectionRate) { this.nonSpamDetectionRate = nonSpamDetectionRate; }

        public int[] getHiddenLayerSizes() { return hiddenLayerSizes; }
        public void setHiddenLayerSizes(int[] hiddenLayerSizes) { this.hiddenLayerSizes = hiddenLayerSizes; }

        public int getNumHiddenLayers() { return numHiddenLayers; }
        public void setNumHiddenLayers(int numHiddenLayers) { this.numHiddenLayers = numHiddenLayers; }

        public String getActivationFunction() { return activationFunction; }
        public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }
    }

    // Classe interne représentant la perte à chaque époque
    public static class EpochLoss {
        private int epoch; // Numéro de l’époque
        private double loss; // Valeur de la perte

        public EpochLoss() {}

        public EpochLoss(int epoch, double loss) {
            this.epoch = epoch;
            this.loss = loss;
        }

        public int getEpoch() { return epoch; }
        public void setEpoch(int epoch) { this.epoch = epoch; }

        public double getLoss() { return loss; }
        public void setLoss(double loss) { this.loss = loss; }
    }

    // Accesseurs principaux pour TrainingResponse
    public TrainingResponse() {}

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public TrainingMetrics getMetrics() { return metrics; }
    public void setMetrics(TrainingMetrics metrics) { this.metrics = metrics; }

    public List<EpochLoss> getEpochLosses() { return epochLosses; }
    public void setEpochLosses(List<EpochLoss> epochLosses) { this.epochLosses = epochLosses; }
}
