package neuralnet_project;

import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.BackPropagation;

/**
 * @author bobboau
 *
 */
public class ExtBackPropigation extends BackPropagation {

	/**
	 *
	 */
	public int num=0;
	private static final long serialVersionUID = 8446616068402387617L;

	double learning_decay_rate = 0.95;
	public void setLearningDecayRate(double _learning_decay_rate){
		learning_decay_rate = _learning_decay_rate;
	}

	double learning_regen_rate = 1.0;
	public void setLearningRegenRate(double _learning_regen_rate){
		learning_regen_rate = _learning_regen_rate;
	}

	double batch_size = 0;
	public void setBatchSize(int _batch_size){
		batch_size = _batch_size;
		if(_batch_size!=0)
			this.setBatchMode(true);
	}

	double batch_size_decay_rate = 1.0;
	public void setBatchSizeDecayRate(double _batch_size_decay_rate){
		batch_size_decay_rate = _batch_size_decay_rate;
	}

	double batch_size_regen_rate = 1.0;
	public void setBatchSizeRegenRate(double _batch_size_regen_rate){
		batch_size_regen_rate = _batch_size_regen_rate;
	}

	/**
	 *
	 */
	public ExtBackPropigation() {
		super();
	}

	double last_error = 1.0;

	@Override
	protected void afterEpoch(){
		super.afterEpoch();
		helperToWrite(this.previousEpochError);
		if(last_error < this.previousEpochError){
			learningRate *= learning_decay_rate;
			batch_size *= batch_size_decay_rate;
			if(isInBatchMode() && batch_size < 1){
				this.setBatchMode(false);
			}
			System.out.println("     learning rate set to: "+this.learningRate+" on epoch "+this.currentIteration+" with error: "+this.previousEpochError);
		}
		else{
			System.out.println("     Error decreaseing and not changed from last on epoch "+this.currentIteration+" with error: "+this.previousEpochError);
			num=this.currentIteration;
			batch_size *= batch_size_regen_rate;
			learningRate *= learning_regen_rate;
			if(!isInBatchMode() && batch_size >= 1){
				this.setBatchMode(true);
			}
		}
		last_error = this.previousEpochError;

	}

	/**
	 * This method implements basic logic for one learning epoch for the
	 * supervised learning algorithms. Epoch is the one pass through the
	 * training set. This method  iterates through the training set
	 * and trains network for each element. It also sets flag if conditions
	 * to stop learning has been reached: network error below some allowed
	 * value, or maximum iteration count
	 *
	 * @param trainingSet training set for training network
	 */
	@Override
	public void doLearningEpoch(DataSet trainingSet) {

		if(this.isInBatchMode() && batch_size >= 1){
			// feed network with all elements from training set
			Iterator<DataSetRow> iterator = trainingSet.iterator();
			ArrayList<DataSetRow> random_ordered_set = new ArrayList<DataSetRow>();
			while (iterator.hasNext() && !isStopped()) {
				random_ordered_set.add(iterator.next());
			}
			/*
			long seed = System.nanoTime();
			Collections.shuffle(random_ordered_set, new Random(seed));
			*/
			int count = 0;
			for( int i = 0; i<random_ordered_set.size(); i++){
				DataSetRow dataSetRow = random_ordered_set.get(i);
				// learn current input/output pattern defined by SupervisedTrainingElement
				this.learnPattern(dataSetRow);
				if(++count >= batch_size){
					count = 0;
					doBatchWeightsUpdate();
				}
			}
			this.totalNetworkError = getErrorFunction().getTotalError();
		}
		else{
			super.doLearningEpoch(trainingSet);
		}
	}
	public static void helperToWrite(Double d){
		try {
//
			DataOutputStream dos = new DataOutputStream(new FileOutputStream("H:/error.txt",true));
				String line = d.toString();
				dos.write(line.getBytes());
				dos.write('\r');
				dos.write('\n');

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}