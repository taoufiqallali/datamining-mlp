package m2i.datamining_mlp.DTO;

public class TrainingRequest {
    private int[] hiddenSizes; // Array of hidden layer sizes
    private String activationFunction; // Activation function name
    private double learningRate;
    private int epochs;

    public TrainingRequest() {}

    public TrainingRequest(int[] hiddenSizes, String activationFunction, double learningRate, int epochs) {
        this.hiddenSizes = hiddenSizes;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    public int[] getHiddenSizes() {
        return hiddenSizes;
    }

    public void setHiddenSizes(int[] hiddenSizes) {
        this.hiddenSizes = hiddenSizes;
    }

    public String getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
}