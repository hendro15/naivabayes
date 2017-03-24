/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.Filter;

/**
 *
 * @author Hendro E. Prabowo
 */
public class Naive {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        String csvFile = "C:\\Users\\sonic_adv\\Downloads\\WISDM_ar_v1.1\\WISDM_ar_v1.1_transformed.arff";
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(csvFile);
        Instances dataset = source.getDataSet();

        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }

        int trainSize = (int) Math.round(dataset.numInstances() * 0.9);
        int testSize = dataset.numInstances() - trainSize;
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);
        
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(naiveBayes, test);
        
        double label = naiveBayes.classifyInstance(test.instance(0));
        test.instance(0).setClassValue(label);
        
        // show prediction result
        System.out.println("Class predicted : " + test.classAttribute().value((int) label));
        
        // get confussion matrix
        double[][] conMat = eval.confusionMatrix();
        
        // show the number of attributes
        System.out.println(eval.toSummaryString("\nResult\n========================\n", false));
        System.out.println(eval.toClassDetailsString("\nClass statistics\n============================================================="));
        System.out.println(eval.toMatrixString("\nConfussion Matrix\n========================================================"));
    }

}
