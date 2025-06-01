package m2i.datamining_mlp.DTO;

public class TrainingRequest {
    private int hiddenSize;
    private double learningRate;
    private int epochs;

    public TrainingRequest() {}

    public TrainingRequest(int hiddenSize, double learningRate, int epochs) {
        this.hiddenSize = hiddenSize;

        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
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