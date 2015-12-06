package neuralnet_project;

import java.io.BufferedReader;
import java.io.IOException;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.samples.convolution.MNISTDataSet;
import org.neuroph.util.TransferFunctionType;

import java.io.InputStreamReader;
import java.util.List;
import java.util.ArrayList;

/**
 * @author bobboau
 *
 */
public class maintest {

    /**
     * @param args
     */
    public static void main(String[] args) {

        try {

            //make the training and test data sets
            DataSet training_set = MNISTDataSet.createFromFile(
                    MNISTDataSet.TRAIN_LABEL_NAME,
                    MNISTDataSet.TRAIN_IMAGE_NAME,
                    50000
            );
            DataSet validation_set = mnistdataread.createFromFile(
                    MNISTDataSet.TRAIN_LABEL_NAME,
                    MNISTDataSet.TRAIN_IMAGE_NAME,
                    10000
            );
            DataSet test_set = MNISTDataSet.createFromFile(
                    MNISTDataSet.TEST_LABEL_NAME,
                    MNISTDataSet.TEST_IMAGE_NAME,
                    10000
            );

            //get a trained neaural net
            double[] learning_rate=new double[7];
            float[] accuracy=new float[7];
            //List<NeuralNetwork<BackPropagation>> neural_net=new ArrayList<NeuralNetwork<BackPropagation>>();
            int i;
            learning_rate[0]=0.002;
            learning_rate[1]=0.003;
            learning_rate[2]=0.004;
            learning_rate[3]=0.006;
            learning_rate[4]=0.007;
            learning_rate[5]=0.008;
            learning_rate[6]=0.009;
            for(i=0;i<7;i++) {
                NeuralNetwork<BackPropagation> net = train(training_set,learning_rate[i]);
                //neural_net.add(net);
                float success = evaluate(net, validation_set);
                System.out.println("Neural net evaluated with "+success*100+"% accuracy.");
                accuracy[i]=success;
            }
            for(i=0;i<7;i++) {
                System.out.println("Learning rate is "+learning_rate[i]+", accuracy is "+accuracy[i]);
            }

            //System.out.println("Testing network.");

            //see how well it does with the test set
            //float success = evaluate(neural_net.get(maxindex), test_set);

            //System.out.println("Neural net evaluated test set with "+success*100+"% accuracy.");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private static NeuralNetwork<BackPropagation> train(DataSet training_set, double learning_rate){
        System.out.println("Making network.");
        NeuralNetwork<BackPropagation> neural_network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, training_set.getInputSize(), 30, training_set.getOutputSize());
        ExtBackPropigation backPropagation = new ExtBackPropigation();
        backPropagation.setMaxIterations(200);
        backPropagation.setLearningRate(learning_rate);
        backPropagation.setMaxError(0.001);
		/*
		backPropagation.setBatchSize(BatchSize);
		backPropagation.setBatchSizeDecayRate(BatchSizeDecayRate);
		backPropagation.setBatchSizeRegenRate(BatchSizeRegenRate);
		backPropagation.setLearningDecayRate(LearningDecayRate);
		backPropagation.setLearningRegenRate(LearningRegenRate);
		*/
        System.out.println("Training network.");
        neural_network.learn(training_set, backPropagation);
        return neural_network;
    }

    private static float evaluate(NeuralNetwork<BackPropagation> neural_net, DataSet test_set){
        int number_right = 0;
        int cur = 0;
        for(DataSetRow data_row : test_set.getRows()) {
            neural_net.setInput(data_row.getInput());
            neural_net.calculate();

            double[] actual_output = neural_net.getOutput();

            //figure out which class is the most likely
            double greatest_probability = -1;
            int greatest_idx = -1;
            for(int i = 0; i<actual_output.length; i++){
                if(greatest_probability < actual_output[i]){
                    greatest_probability = actual_output[i];
                    greatest_idx = i;
                }
            }

            //check against desired
            double[] desired_output = data_row.getDesiredOutput();
            boolean is_correct = true;
            for(int i = 0; i<desired_output.length; i++){
                if((i == greatest_idx) != (desired_output[i] == 1.0)){
                    is_correct = false;
                }
            }
            if(is_correct){
                number_right++;
            }

            System.out.println("     Evaluating... "+(((float)++cur)/((float)test_set.getRows().size())*100)+"%, so far it's looking like "+(((float)number_right)/((float)cur)*100)+"%");
        }
        return ((float)number_right)/((float)test_set.getRows().size());
    }
}