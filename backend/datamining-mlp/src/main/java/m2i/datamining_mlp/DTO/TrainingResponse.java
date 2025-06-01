package m2i.datamining_mlp.DTO;

import java.util.List;

public class TrainingResponse {
    private String status;
    private String message;
    private TrainingMetrics metrics;
    private List<EpochLoss> epochLosses;

    public static class TrainingMetrics {
        private int totalEmails;
        private int spamEmails;
        private int nonSpamEmails;
        private int featureDimensions;
        private int trainSize;
        private int testSize;
        private double accuracy;
        private double spamDetectionRate;
        private double nonSpamDetectionRate;

        // Constructors, getters and setters
        public TrainingMetrics() {}

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
    }

    public static class EpochLoss {
        private int epoch;
        private double loss;

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

    // Main class getters and setters
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
