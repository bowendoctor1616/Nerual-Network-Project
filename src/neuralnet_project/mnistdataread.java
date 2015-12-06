package neuralnet_project;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.samples.convolution.MNISTImage;
import org.neuroph.samples.convolution.MNISTImageLoader;

import java.io.IOException;
import java.util.List;

/**
 * Created by Bowen on 2015/12/4.
 */
public class mnistdataread {
    public static final String TRAIN_LABEL_NAME = "train-labels.idx1-ubyte";
    public static final String TRAIN_IMAGE_NAME = "train-images.idx3-ubyte";
    public static final String TEST_LABEL_NAME = "t10k-labels.idx1-ubyte";
    public static final String TEST_IMAGE_NAME = "t10k-images.idx3-ubyte";

    public mnistdataread() {
    }

    public static DataSet createFromFile(String labelPath, String imagePath, int sampleCount) throws IOException {
        MNISTImageLoader mnistLoader = new MNISTImageLoader("/" + labelPath, "/" + imagePath);
        List mnistImages = mnistLoader.loadDigitImages();
        DataSet dataSet = createDataSet(mnistImages, sampleCount);
        return dataSet;
    }

    private static DataSet createDataSet(List<MNISTImage> imageList, int sampleCount) {
        int pixelCount = ((MNISTImage)imageList.get(0)).getSize();
        DataSet dataSet = new DataSet(pixelCount, 10);

        for(int i = 0; i < sampleCount; ++i) {
            MNISTImage dImage = (MNISTImage)imageList.get(i+50000);
            double[] input = new double[pixelCount];
            double[] output = new double[10];
            output[dImage.getLabel()] = 1.0D;
            byte[] imageData = dImage.getData();

            for(int row = 0; row < pixelCount; ++row) {
                input[row] = (double)(imageData[row] & 255) / 255.0D;
            }

            DataSetRow var10 = new DataSetRow(input, output);
            dataSet.addRow(var10);
        }

        return dataSet;
    }
}
