package m2i.datamining_mlp.service;

import m2i.datamining_mlp.DTO.TrainingRequest;
import m2i.datamining_mlp.DTO.TrainingResponse;
import m2i.datamining_mlp.model.Classifier;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Service class for training and using a neural network classifier for email spam detection.
 * Handles dataset loading, model training, prediction, and retrieval of model metrics.
 */
@Service
public class ClassifierService {

    /** The currently trained classifier instance. */
    private Classifier currentClassifier;

    /** Stores the metrics from the last training session. */
    private TrainingResponse.TrainingMetrics lastTrainingMetrics;

    /**
     * Trains a neural network classifier using the provided configuration and dataset.
     *
     * @param request The training request containing hidden layer sizes, activation function,
     *                learning rate, and number of epochs.
     * @return A TrainingResponse object containing the training status, message, metrics,
     *         and epoch loss history.
     */
    public TrainingResponse trainModel(TrainingRequest request) {
        TrainingResponse response = new TrainingResponse();
        List<TrainingResponse.EpochLoss> epochLosses = new ArrayList<>();

        try {
            // Validate hidden layer sizes
            if (request.getHiddenSizes() == null || request.getHiddenSizes().length == 0) {
                response.setStatus("error");
                response.setMessage("Hidden layer sizes must be specified");
                return response;
            }

            // Ensure all hidden layer sizes are positive
            for (int size : request.getHiddenSizes()) {
                if (size <= 0) {
                    response.setStatus("error");
                    response.setMessage("All hidden layer sizes must be positive");
                    return response;
                }
            }

            // Parse and validate activation function
            Classifier.ActivationFunction activationFunction;
            try {
                activationFunction = Classifier.ActivationFunction.valueOf(request.getActivationFunction().toUpperCase());
            } catch (IllegalArgumentException e) {
                response.setStatus("error");
                response.setMessage("Invalid activation function. Valid options: SIGMOID, TANH, RELU, LEAKY_RELU");
                return response;
            }

            // Load dataset from CSV file
            List<String[]> dataset = loadDataset();

            // Check if dataset is valid
            if (dataset.isEmpty()) {
                response.setStatus("error");
                response.setMessage("Dataset is empty or invalid");
                return response;
            }

            // Determine feature count (excluding first column ID and last column target)
            String[] header = dataset.get(0);
            int featureCount = header.length - 2; // -2 to exclude ID and target columns

            // Validate feature count
            if (featureCount <= 0) {
                response.setStatus("error");
                response.setMessage("No valid features found in dataset");
                return response;
            }

            // Initialize arrays for features and targets
            double[][] features = new double[dataset.size() - 1][featureCount]; // -1 to skip header
            int[] target = new int[dataset.size() - 1];

            // Parse dataset, skipping first column (ID) and extracting last column as target
            for (int i = 1; i < dataset.size(); i++) { // Start from 1 to skip header
                String[] row = dataset.get(i);

                // Parse features, starting from index 1 to skip ID
                for (int j = 0; j < featureCount; j++) {
                    try {
                        features[i - 1][j] = Double.parseDouble(row[j + 1]);
                    } catch (NumberFormatException e) {
                        features[i - 1][j] = 0.0; // Default to 0 for invalid numbers
                    }
                }

                // Parse target from the last column
                try {
                    target[i - 1] = Integer.parseInt(row[header.length - 1]);
                } catch (NumberFormatException e) {
                    target[i - 1] = 0; // Default to 0 for invalid target
                }
            }

            // Calculate dataset statistics
            int totalEmails = features.length;
            int spamCount = 0;
            for (int t : target) {
                if (t == 1) spamCount++;
            }

            // Split dataset into training (80%) and test (20%) sets
            int trainSize = (int) (totalEmails * 0.8);
            int testSize = totalEmails - trainSize;

            // Shuffle indices for random train-test split
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < totalEmails; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, new Random(42));

            // Initialize training and test sets
            double[][] xTrain = new double[trainSize][featureCount];
            double[][] xTest = new double[testSize][featureCount];
            int[] yTrain = new int[trainSize];
            int[] yTest = new int[testSize];

            // Populate training set
            for (int i = 0; i < trainSize; i++) {
                xTrain[i] = features[indices.get(i)].clone();
                yTrain[i] = target[indices.get(i)];
            }

            // Populate test set
            for (int i = 0; i < testSize; i++) {
                xTest[i] = features[indices.get(i + trainSize)].clone();
                yTest[i] = target[indices.get(i + trainSize)];
            }

            // Initialize classifier with specified architecture
            currentClassifier = new Classifier(featureCount, request.getHiddenSizes(),
                    request.getLearningRate(), activationFunction);

            // Train the model, tracking loss per epoch
            for (int epoch = 0; epoch < request.getEpochs(); epoch++) {
                double totalLoss = 0.0;

                // Shuffle training data for each epoch
                List<Integer> trainIndices = new ArrayList<>();
                for (int i = 0; i < trainSize; i++) {
                    trainIndices.add(i);
                }
                Collections.shuffle(trainIndices);

                // Train on each sample and compute loss
                for (int idx : trainIndices) {
                    currentClassifier.trainSample(xTrain[idx], yTrain[idx]);

                    double prediction = currentClassifier.predict(xTrain[idx]);
                    double loss = Math.pow(yTrain[idx] - prediction, 2);
                    totalLoss += loss;
                }

                // Record average loss for every 10th epoch or the last epoch
                double avgLoss = totalLoss / xTrain.length;
                if (epoch % 10 == 0 || epoch == request.getEpochs() - 1) {
                    epochLosses.add(new TrainingResponse.EpochLoss(epoch, avgLoss));
                }
            }

            // Evaluate model on test set
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

            // Build response metrics
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

            // Set neural network architecture information
            metrics.setHiddenLayerSizes(request.getHiddenSizes());
            metrics.setNumHiddenLayers(request.getHiddenSizes().length);
            metrics.setActivationFunction(request.getActivationFunction());

            // Store metrics for later retrieval
            lastTrainingMetrics = metrics;

            // Set successful response
            response.setStatus("success");
            response.setMessage(String.format("Model trained successfully with %d hidden layers using %s activation",
                    request.getHiddenSizes().length, request.getActivationFunction()));
            response.setMetrics(metrics);
            response.setEpochLosses(epochLosses);

        } catch (Exception e) {
            response.setStatus("error");
            response.setMessage("Training failed: " + e.getMessage());
            e.printStackTrace(); // TODO: Replace with proper logging in production
        }

        return response;
    }

    /**
     * Predicts whether an email is spam based on its features.
     *
     * @param features The feature vector of the email (excluding ID).
     * @return A map containing the prediction result, including the raw prediction score,
     *         spam classification, confidence, and model information.
     */
    public Map<String, Object> predictEmail(double[] features) {
        Map<String, Object> result = new HashMap<>();

        // Check if a trained model exists
        if (currentClassifier == null) {
            result.put("error", "No trained model available");
            return result;
        }

        try {
            // Validate feature vector size
            if (features.length != currentClassifier.getInputSize()) {
                result.put("error", String.format("Feature vector size mismatch. Expected %d, got %d",
                        currentClassifier.getInputSize(), features.length));
                return result;
            }

            // Make prediction
            double prediction = currentClassifier.predict(features);
            boolean isSpam = prediction > 0.5;
            double confidence = isSpam ? prediction : (1 - prediction);

            // Populate result
            result.put("prediction", prediction);
            result.put("isSpam", isSpam);
            result.put("classification", isSpam ? "SPAM" : "NOT SPAM");
            result.put("confidence", confidence);
            result.put("modelInfo", String.format("Network: %d layers, %s activation",
                    currentClassifier.getNumHiddenLayers(),
                    currentClassifier.getActivationFunction()));

        } catch (Exception e) {
            result.put("error", "Prediction failed: " + e.getMessage());
        }

        return result;
    }

    /**
     * Retrieves the metrics from the last training session.
     *
     * @return The TrainingMetrics object containing details of the last training, or null if no training has occurred.
     */
    public TrainingResponse.TrainingMetrics getLastTrainingMetrics() {
        return lastTrainingMetrics;
    }

    /**
     * Retrieves information about the currently trained model.
     *
     * @return A map containing model details such as input size, hidden layer sizes,
     *         number of hidden layers, activation function, and learning rate.
     */
    public Map<String, Object> getModelInfo() {
        Map<String, Object> info = new HashMap<>();

        // Check if a trained model exists
        if (currentClassifier == null) {
            info.put("error", "No trained model available");
            return info;
        }

        // Populate model information
        info.put("inputSize", currentClassifier.getInputSize());
        info.put("hiddenLayerSizes", currentClassifier.getHiddenSizes());
        info.put("numHiddenLayers", currentClassifier.getNumHiddenLayers());
        info.put("activationFunction", currentClassifier.getActivationFunction().toString());
        info.put("learningRate", currentClassifier.getLearningRate());

        return info;
    }

    /**
     * Loads the email dataset from a CSV file.
     *
     * @return A list of string arrays, where each array represents a row in the CSV file.
     * @throws Exception If an error occurs while reading the file.
     */
    private List<String[]> loadDataset() throws Exception {
        List<String[]> dataset = new ArrayList<>();
        String datasetPath = "src/main/resources/dataset/emails.csv";

        try (BufferedReader br = new BufferedReader(new FileReader(datasetPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Parse CSV line, handling quoted fields
                String[] row = parseCsvLine(line);
                dataset.add(row);
            }
        }

        return dataset;
    }

    /**
     * Parses a CSV line, handling quoted fields that may contain commas.
     *
     * @param line The CSV line to parse.
     * @return An array of strings representing the fields in the line.
     */
    private String[] parseCsvLine(String line) {
        List<String> result = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder field = new StringBuilder();

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                result.add(field.toString().trim());
                field = new StringBuilder();
            } else {
                field.append(c);
            }
        }

        // Add the last field
        result.add(field.toString().trim());
        return result.toArray(new String[0]);
    }
}
