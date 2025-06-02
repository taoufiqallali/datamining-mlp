package m2i.datamining_mlp.model;

import java.util.Random;

public class Classifier {
    // Network architecture
    private int inputSize;
    private int[] hiddenSizes;  // Array to store sizes of each hidden layer
    private int outputSize;
    private int numHiddenLayers;

    // Activation function enum
    public enum ActivationFunction {
        SIGMOID, TANH, RELU, LEAKY_RELU
    }

    private ActivationFunction activationFunction;

    // Weights and biases - now arrays for multiple layers
    private double[][][] weights;  // [layer][from][to]
    private double[][] biases;     // [layer][neuron]

    // Learning rate
    private double learningRate;

    // Random generator
    private Random random;

    public Classifier(int inputSize, int[] hiddenSizes, double learningRate, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes.clone();
        this.numHiddenLayers = hiddenSizes.length;
        this.outputSize = 1; // Binary classification (spam or not)
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.random = new Random(42); // Fixed seed

        initializeWeights();
    }

    /**
     * Initialize weights and biases with small random values
     */
    private void initializeWeights() {
        // Total layers = hidden layers + output layer
        int totalLayers = numHiddenLayers + 1;
        weights = new double[totalLayers][][];
        biases = new double[totalLayers][];

        // Initialize weights and biases for each layer
        for (int layer = 0; layer < totalLayers; layer++) {
            int inputSizeForLayer, outputSizeForLayer;

            if (layer == 0) {
                // First hidden layer
                inputSizeForLayer = inputSize;
                outputSizeForLayer = hiddenSizes[0];
            } else if (layer < numHiddenLayers) {
                // Subsequent hidden layers
                inputSizeForLayer = hiddenSizes[layer - 1];
                outputSizeForLayer = hiddenSizes[layer];
            } else {
                // Output layer
                inputSizeForLayer = hiddenSizes[numHiddenLayers - 1];
                outputSizeForLayer = outputSize;
            }

            // Initialize weights for this layer
            weights[layer] = new double[inputSizeForLayer][outputSizeForLayer];
            for (int i = 0; i < inputSizeForLayer; i++) {
                for (int j = 0; j < outputSizeForLayer; j++) {
                    weights[layer][i][j] = random.nextGaussian() * 0.1;
                }
            }

            // Initialize biases for this layer
            biases[layer] = new double[outputSizeForLayer];
            for (int i = 0; i < outputSizeForLayer; i++) {
                biases[layer][i] = random.nextGaussian() * 0.1;
            }
        }
    }

    /**
     * Apply activation function
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
                return 1.0 / (1.0 + Math.exp(-x)); // Default to sigmoid
        }
    }

    /**
     * Calculate derivative of activation function
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
     * Forward propagation
     * @param input Input features (email word frequencies)
     * @return Prediction probability (0-1, where >0.5 means spam)
     */
    public double predict(double[] input) {
        double[] currentInput = input.clone();

        // Forward through all layers
        for (int layer = 0; layer < numHiddenLayers + 1; layer++) {
            double[] nextLayer = new double[weights[layer][0].length];

            for (int j = 0; j < nextLayer.length; j++) {
                double sum = biases[layer][j];
                for (int i = 0; i < currentInput.length; i++) {
                    sum += currentInput[i] * weights[layer][i][j];
                }

                // Apply activation function (sigmoid for output layer, user-defined for hidden layers)
                if (layer < numHiddenLayers) {
                    nextLayer[j] = activate(sum, activationFunction);
                } else {
                    nextLayer[j] = activate(sum, ActivationFunction.SIGMOID); // Always sigmoid for output
                }
            }

            currentInput = nextLayer;
        }

        return currentInput[0]; // Return the single output
    }

    /**
     * Train the network on one sample (single email)
     * @param input Email features
     * @param target True label (0=not spam, 1=spam)
     */
    public void trainSample(double[] input, int target) {
        // FORWARD PASS - store intermediate values for backprop
        double[][] layerOutputs = new double[numHiddenLayers + 2][]; // +2 for input and output
        double[][] layerInputs = new double[numHiddenLayers + 1][]; // +1 for hidden and output layers

        layerOutputs[0] = input.clone(); // Input layer

        // Forward through all layers
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

                // Apply activation function
                if (layer < numHiddenLayers) {
                    layerOutputs[layer + 1][j] = activate(sum, activationFunction);
                } else {
                    layerOutputs[layer + 1][j] = activate(sum, ActivationFunction.SIGMOID);
                }
            }
        }

        // BACKWARD PASS - calculate gradients
        double[][] deltas = new double[numHiddenLayers + 1][];

        // Calculate delta for output layer
        int outputLayerIndex = numHiddenLayers;
        deltas[outputLayerIndex] = new double[outputSize];
        double output = layerOutputs[outputLayerIndex + 1][0];
        double outputError = target - output;
        deltas[outputLayerIndex][0] = outputError * output * (1 - output); // Sigmoid derivative

        // Calculate deltas for hidden layers (backpropagate)
        for (int layer = numHiddenLayers - 1; layer >= 0; layer--) {
            deltas[layer] = new double[hiddenSizes[layer]];

            for (int i = 0; i < hiddenSizes[layer]; i++) {
                double error = 0.0;

                // Sum errors from next layer
                for (int j = 0; j < deltas[layer + 1].length; j++) {
                    error += deltas[layer + 1][j] * weights[layer + 1][i][j];
                }

                // Apply derivative of activation function
                deltas[layer][i] = error * activationDerivative(layerInputs[layer][i], activationFunction);
            }
        }

        // UPDATE WEIGHTS AND BIASES
        for (int layer = 0; layer < numHiddenLayers + 1; layer++) {
            // Update weights
            for (int i = 0; i < weights[layer].length; i++) {
                for (int j = 0; j < weights[layer][i].length; j++) {
                    weights[layer][i][j] += learningRate * deltas[layer][j] * layerOutputs[layer][i];
                }
            }

            // Update biases
            for (int j = 0; j < biases[layer].length; j++) {
                biases[layer][j] += learningRate * deltas[layer][j];
            }
        }
    }

    // Getters for network parameters
    public int getInputSize() { return inputSize; }
    public int[] getHiddenSizes() { return hiddenSizes.clone(); }
    public int getNumHiddenLayers() { return numHiddenLayers; }
    public ActivationFunction getActivationFunction() { return activationFunction; }
    public double getLearningRate() { return learningRate; }
}