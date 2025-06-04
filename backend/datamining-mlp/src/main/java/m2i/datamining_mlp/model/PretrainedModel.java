package m2i.datamining_mlp.model;


import m2i.datamining_mlp.DTO.TrainingResponse;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.io.Serializable;
import java.util.Arrays;

@Document(collection = "pretrained_models")
public class PretrainedModel implements Serializable {
    @Id
    private String id; // Fixed ID for single pretrained model
    private double[][][] weights; // Neural network weights
    private double[][] biases; // Neural network biases
    private int inputSize;
    private int[] hiddenSizes;
    private double learningRate;
    private String activationFunction;
    private TrainingResponse.TrainingMetrics metrics;

    // Constructors
    public PretrainedModel() {}

    public PretrainedModel(Classifier classifier, TrainingResponse.TrainingMetrics metrics) {
        this.weights = classifier.getWeights();
        this.biases = classifier.getBiases();
        this.inputSize = classifier.getInputSize();
        this.hiddenSizes = classifier.getHiddenSizes();
        this.learningRate = classifier.getLearningRate();
        this.activationFunction = classifier.getActivationFunction().toString();
        this.metrics = metrics;
    }

    // Getters and Setters
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

    // Convert to Classifier
    public Classifier toClassifier() {
        Classifier classifier = new Classifier(inputSize, hiddenSizes, learningRate,
                Classifier.ActivationFunction.valueOf(activationFunction));
        classifier.setWeights(weights);
        classifier.setBiases(biases);
        return classifier;
    }
}

