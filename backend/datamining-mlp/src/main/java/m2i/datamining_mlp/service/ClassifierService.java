package m2i.datamining_mlp.service;

import m2i.datamining_mlp.DTO.TrainingRequest;
import m2i.datamining_mlp.DTO.TrainingResponse;
import m2i.datamining_mlp.model.Classifier;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.*;

@Service
public class ClassifierService {

    private Classifier currentClassifier;
    private TrainingResponse.TrainingMetrics lastTrainingMetrics;

    public TrainingResponse trainModel(TrainingRequest request) {
        TrainingResponse response = new TrainingResponse();
        List<TrainingResponse.EpochLoss> epochLosses = new ArrayList<>();

        try {
            // Load and parse dataset
            List<String[]> dataset = loadDataset();

            if (dataset.isEmpty()) {
                response.setStatus("error");
                response.setMessage("Dataset is empty or invalid");
                return response;
            }

            // Get header to determine feature count
            String[] header = dataset.get(0);
            int featureCount = header.length - 1; // -1 for target column

            // Convert data to numerical format
            double[][] features = new double[dataset.size() - 1][featureCount]; // -1 to skip header
            int[] target = new int[dataset.size() - 1];

            for (int i = 1; i < dataset.size(); i++) { // Start from 1 to skip header
                String[] row = dataset.get(i);

                // Parse features
                for (int j = 0; j < featureCount; j++) {
                    try {
                        features[i-1][j] = Double.parseDouble(row[j]);
                    } catch (NumberFormatException e) {
                        features[i-1][j] = 0.0;
                    }
                }

                // Parse target
                try {
                    target[i-1] = Integer.parseInt(row[featureCount]);
                } catch (NumberFormatException e) {
                    target[i-1] = 0;
                }
            }

            // Calculate dataset statistics
            int totalEmails = features.length;
            int spamCount = 0;
            for (int t : target) {
                if (t == 1) spamCount++;
            }

            // Split data
            int trainSize = (int) (totalEmails * 0.8);
            int testSize = totalEmails - trainSize;

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < totalEmails; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, new Random(42));

            double[][] xTrain = new double[trainSize][featureCount];
            double[][] xTest = new double[testSize][featureCount];
            int[] yTrain = new int[trainSize];
            int[] yTest = new int[testSize];

            for (int i = 0; i < trainSize; i++) {
                xTrain[i] = features[indices.get(i)].clone();
                yTrain[i] = target[indices.get(i)];
            }

            for (int i = 0; i < testSize; i++) {
                xTest[i] = features[indices.get(i + trainSize)].clone();
                yTest[i] = target[indices.get(i + trainSize)];
            }

            // Create and train classifier
            currentClassifier = new Classifier(featureCount, request.getHiddenSize(), request.getLearningRate());

            // Custom training with epoch tracking
            for (int epoch = 0; epoch < request.getEpochs(); epoch++) {
                double totalLoss = 0.0;

                for (int i = 0; i < xTrain.length; i++) {
                    currentClassifier.trainSample(xTrain[i], yTrain[i]);

                    double prediction = currentClassifier.predict(xTrain[i]);
                    double loss = Math.pow(yTrain[i] - prediction, 2);
                    totalLoss += loss;
                }

                double avgLoss = totalLoss / xTrain.length;
                if (epoch % 10 == 0) {
                    epochLosses.add(new TrainingResponse.EpochLoss(epoch, avgLoss));
                }
            }

            // Evaluate model
            int correct = 0;
            int totalSpam = 0;
            int correctSpam = 0;
            int totalNotSpam = 0;
            int correctNotSpam = 0;

            for (int i = 0; i < xTest.length; i++) {
                double prediction = currentClassifier.predict(xTest[i]);
                int predictedClass = prediction > 0.5 ? 1 : 0;

                if (predictedClass == yTest[i]) {
                    correct++;
                }

                if (yTest[i] == 1) {
                    totalSpam++;
                    if (predictedClass == 1) correctSpam++;
                } else {
                    totalNotSpam++;
                    if (predictedClass == 0) correctNotSpam++;
                }
            }

            // Build response
            TrainingResponse.TrainingMetrics metrics = new TrainingResponse.TrainingMetrics();
            metrics.setTotalEmails(totalEmails);
            metrics.setSpamEmails(spamCount);
            metrics.setNonSpamEmails(totalEmails - spamCount);
            metrics.setFeatureDimensions(featureCount);
            metrics.setTrainSize(trainSize);
            metrics.setTestSize(testSize);
            metrics.setAccuracy((double) correct / testSize);
            metrics.setSpamDetectionRate(totalSpam > 0 ? (double) correctSpam / totalSpam : 0);
            metrics.setNonSpamDetectionRate(totalNotSpam > 0 ? (double) correctNotSpam / totalNotSpam : 0);

            lastTrainingMetrics = metrics;

            response.setStatus("success");
            response.setMessage("Model trained successfully");
            response.setMetrics(metrics);
            response.setEpochLosses(epochLosses);

        } catch (Exception e) {
            response.setStatus("error");
            response.setMessage("Training failed: " + e.getMessage());
        }

        return response;
    }

    public Map<String, Object> predictEmail(double[] features) {
        Map<String, Object> result = new HashMap<>();

        if (currentClassifier == null) {
            result.put("error", "No trained model available");
            return result;
        }

        try {
            double prediction = currentClassifier.predict(features);
            boolean isSpam = prediction > 0.5;
            double confidence = isSpam ? prediction : (1 - prediction);

            result.put("prediction", prediction);
            result.put("isSpam", isSpam);
            result.put("classification", isSpam ? "SPAM" : "NOT SPAM");
            result.put("confidence", confidence);

        } catch (Exception e) {
            result.put("error", "Prediction failed: " + e.getMessage());
        }

        return result;
    }

    public TrainingResponse.TrainingMetrics getLastTrainingMetrics() {
        return lastTrainingMetrics;
    }

    private List<String[]> loadDataset() throws Exception {
        List<String[]> dataset = new ArrayList<>();
        String datasetPath = "src/main/resources/dataset/emails.csv";

        try (BufferedReader br = new BufferedReader(new FileReader(datasetPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                dataset.add(row);
            }
        }

        return dataset;
    }
}