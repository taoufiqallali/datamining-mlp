package m2i.datamining_mlp.model;

import java.util.Random;

public class Classifier {
    // Network architecture
    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    // Weights and biases
    private double[][] weightsInputHidden;  // inputSize x hiddenSize
    private double[] biasesHidden;          // hiddenSize
    private double[][] weightsHiddenOutput; // hiddenSize x outputSize
    private double[] biasesOutput;          // outputSize

    // Learning rate
    private double learningRate;

    // Random generator
    private Random random;

    public Classifier(int inputSize, int hiddenSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = 1; // Binary classification (spam or not)
        this.learningRate = learningRate;
        this.random = new Random(42); // Fixed seed

        initializeWeights();
    }

    /**
     * Initialize weights and biases with small random values
     */
    private void initializeWeights() {
        // Initialize weights input -> hidden
        weightsInputHidden = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = random.nextGaussian() * 0.1; // Small random values
            }
        }

        // Initialize biases for hidden layer
        biasesHidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            biasesHidden[i] = random.nextGaussian() * 0.1;
        }

        // Initialize weights hidden -> output
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = random.nextGaussian() * 0.1;
            }
        }

        // Initialize biases for output layer
        biasesOutput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            biasesOutput[i] = random.nextGaussian() * 0.1;
        }
    }

    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Forward propagation
     * @param input Input features (email word frequencies)
     * @return Prediction probability (0-1, where >0.5 means spam)
     */
    public double predict(double[] input) {
        // Input to hidden layer
        double[] hiddenLayer = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasesHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            hiddenLayer[j] = sigmoid(sum);
        }

        // Hidden to output layer
        double output = biasesOutput[0];
        for (int i = 0; i < hiddenSize; i++) {
            output += hiddenLayer[i] * weightsHiddenOutput[i][0];
        }

        return sigmoid(output);
    }

    /**
     * Train the network on one sample (single email)
     * @param input Email features
     * @param target True label (0=not spam, 1=spam)
     */
    public void trainSample(double[] input, int target) {
        // FORWARD PASS - store intermediate values for backprop

        // Input to hidden layer
        double[] hiddenLayer = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasesHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            hiddenLayer[j] = sigmoid(sum);
        }

        // Hidden to output layer
        double outputSum = biasesOutput[0];
        for (int i = 0; i < hiddenSize; i++) {
            outputSum += hiddenLayer[i] * weightsHiddenOutput[i][0];
        }
        double output = sigmoid(outputSum);

        // BACKWARD PASS - calculate gradients

        // Output layer error
        double outputError = target - output;
        double outputDelta = outputError * output * (1 - output); // sigmoid derivative

        // Hidden layer errors
        double[] hiddenErrors = new double[hiddenSize];
        double[] hiddenDeltas = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            hiddenErrors[i] = outputDelta * weightsHiddenOutput[i][0];
            hiddenDeltas[i] = hiddenErrors[i] * hiddenLayer[i] * (1 - hiddenLayer[i]);
        }

        // UPDATE WEIGHTS AND BIASES

        // Update hidden to output weights
        for (int i = 0; i < hiddenSize; i++) {
            weightsHiddenOutput[i][0] += learningRate * outputDelta * hiddenLayer[i];
        }

        // Update output bias
        biasesOutput[0] += learningRate * outputDelta;

        // Update input to hidden weights
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * input[i];
            }
        }

        // Update hidden biases
        for (int j = 0; j < hiddenSize; j++) {
            biasesHidden[j] += learningRate * hiddenDeltas[j];
        }
    }

    // Getters for network parameters
    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    public double getLearningRate() { return learningRate; }
}
