package neuralnet_project;

import java.io.*;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.samples.convolution.MNISTDataSet;
import org.neuroph.util.TransferFunctionType;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;

/**
 * @author bobboau
 *
 */
public class test {
    public static final int WEIGHT_NUM = 23860;
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
            double[] learning_rate=new double[33];
            int[] batch=new int[9];
            float[] accuracy=new float[9];
            int i;
            //batch[0]=0;
/*
            for(i=1;i<20;i++)
                batch[i]=i+1;
            for(i=20;i<26;i++)
                batch[i]=(i-15)*5;

            batch[1]=5;
            batch[2]=10;
            batch[3]=35;
            for(i=0;i<10;i++)
                learning_rate[i]=0.001*(i+1);
            for(i=10;i<33;i++)
                learning_rate[i]=0.01*(i-8);
            batch[0]=0;
            for(i=1;i<5;i++)
                batch[i]=5*i;
            for(i=5;i<9;i++)
                batch[i]=5*(i+1);*/
            for(i=0;i<1;i++) {
                NeuralNetwork<BackPropagation> net = train(validation_set,0.004,50);
                //neural_net.add(net);
                float success = evaluate(net, test_set);
                System.out.println("Neural net evaluated with "+success*100+"% accuracy.");
                accuracy[i]=success;
            }
            for(i=0;i<1;i++) {
                System.out.println("Learning rate is "+batch[i]+", accuracy is "+accuracy[i]);
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

    private static NeuralNetwork<BackPropagation> train(DataSet training_set, double learning_rate,int BatchSize){
        System.out.println("Making network.");
        NeuralNetwork<BackPropagation> neural_network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, training_set.getInputSize(), 30, training_set.getOutputSize());
        ExtBackPropigation backPropagation = new ExtBackPropigation();
        backPropagation.setMaxIterations(500);
        backPropagation.setLearningRate(learning_rate);
        backPropagation.setMaxError(0.01);
        backPropagation.setBatchSize(BatchSize);
        backPropagation.setBatchSizeDecayRate(1);
        backPropagation.setBatchSizeRegenRate(1);
        backPropagation.setLearningDecayRate(0.95);
        backPropagation.setLearningRegenRate(1);
        System.out.println("Training network.");
        //        helperToWrite(neural_network.getWeights());
        double[] weights = helperToRead();

        neural_network.setWeights(weights);
        neural_network.learn(training_set, backPropagation);
        helperToWrite((double)backPropagation.num);
        return neural_network;
    }
    public static void helperToWrite(Double d){

        try {
//
            DataOutputStream dos = new DataOutputStream(new FileOutputStream("H:/error.txt",true));

            //write data line by line
            String line = d.toString();
            dos.write(line.getBytes());
            dos.write('\n');

        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
    public static double[] helperToRead(){
        double[] copy = new double[maintest.WEIGHT_NUM];
        try {
            FileReader r = new FileReader("H:/weight");
            BufferedReader br = new BufferedReader(r);
            String line = null;
            int i = 0;
            while(((line = br.readLine())) != null){
                copy[i] = Double.valueOf(line);
                i ++;
            }
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return copy;
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

            System.out.println("     Evaluating.... "+(((float)++cur)/((float)test_set.getRows().size())*100)+"%, so far it's looking like "+(((float)number_right)/((float)cur)*100)+"%");
        }
        return ((float)number_right)/((float)test_set.getRows().size());
    }
}
