import java.util.Scanner;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Pranav Varmaraja
 * 4/21/2022
 * 
 * This class creates, trains, and runs a basic four activation layer perceptron. The network takes any number of inputs, 
 * passes them through 2 activation layers consisting of any number of nodes, and outputs any number of values. The network
 * is thus an A-B-C-D network, with any number of inputs, hidden nodes in layer 1, hidden nodes in layer 2 and outputs.
 * Training is performed via gradient descent to fit the network to training data.
 * 
 * The network also implements backpropagation optimization.
 * 
 * It is configured via various json files: a main config file, a truth table file, a file containing possible inputs, and potentially a
 * file with weights to be used.
 * 
 * All JSON was parsed using JSON-simple, the jar can be found at: https://code.google.com/archive/p/json-simple/downloads
 * 
 * Methods contained in file:
 * 
 * public Network(String fileName)
 * public void parseConfigFile(String fileName)
 * public void parseWeightsFile(String fileName) throws Exception
 * public void parseTruthTable(String fileName)
 * public void parseInputs(String fileName)
 * public void saveWeights(String fileName)
 * public void inputMode()
 * public void setConfigValues()
 * public void echoConfigValues()
 * public void allocate()
 * public void populate()
 * public double getRandWeight()
 * public void randomInitialization()
 * public double activationFunction(double x)
 * public double derivActivationFuncton(double x)
 * public void runNetwork(double[] input)
 * public void executeNetwork(double[] input)
 * public void train() throws Exception
 * public void updateWeights()
 * public void run()
 * public void report()
 * public void execute()
 * public static void main(String[] args)
 */
public class Network 
{

   double[][] weights0;
   double[][] weights1;
   double[][] weights2;
   double[] a;
   double[] h1;
   double[] h2;
   double[] f;
   double[] t;

   enum mode {RUN, TRAIN};
   mode runTrain;

   int numInputs;
   int numOutputs;
   int numHiddenNodes1;
   int numHiddenNodes2;
   double maxErrorThreshold;
   double lambda;
   int maxIterations;
   double maxRandomWeight;
   double minRandomWeight;
   int numPossibleInputSets;
   int numIterations;
   double error;
   double trainingTime;

   double[][] possibleInputs;
   double[][] truthTable;

   double[] thetak;
   double[] thetaj;
   double[] thetai;
   double[] omegak;
   double[] omegaj;
   double[] omegai;
   double[] psik;
   double[] psij;
   double[] psii;

   String outputFileName;
   String weightFileName;
   String inputFileName;
   String truthTableFileName;
   boolean preLoadedWeights;


/**
 * constructor for the Network, sets the config values, echoes then, allocates and populates all instance variable arrays
 */
   public Network(String fileName) 
   {
      parseConfigFile(fileName);
      echoConfigValues();
      allocate();
      populate();
   }

/**
 * parses the json config file using simple-json, assigns config values from json to class variables
 */
   public void parseConfigFile(String fileName)
   {
      Object obj = null;
      boolean runOnly;
      JSONObject json = null;

      try
      {
         obj = new JSONParser().parse(new FileReader(fileName));
      } 
      catch (Exception e)
      {
         e.printStackTrace();
      }
      json = (JSONObject) obj;
      preLoadedWeights = (boolean) json.get("preLoadedWeights");
      runOnly = (boolean) json.get("runOnly");

      numInputs = Math.toIntExact((long) json.get("numInputs"));
      numHiddenNodes1 = Math.toIntExact((long) json.get("numHiddenNodes1"));
      numHiddenNodes2 = Math.toIntExact((long) json.get("numHiddenNodes2"));
      numOutputs = Math.toIntExact((long) json.get("numOutputs"));

      if (!preLoadedWeights)
      {
         minRandomWeight = (double) json.get("minRandomWeight");
         maxRandomWeight = (double) json.get("maxRandomWeight");
      }
      else 
      {
         try
         {
            weightFileName = ((String) json.get("weightFileName"));
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(0);
         }
      } //else

      if (runOnly)
      {
         runTrain = mode.RUN;
      }
      else
      {
         runTrain = mode.TRAIN;
         maxIterations = Math.toIntExact((long) json.get("maxIterations"));
         outputFileName = (String) json.get("outputWeightsFileName");
         maxErrorThreshold = (double) json.get("errorThreshold");
         lambda = (double) json.get("lambda");         
      } //else

      truthTableFileName = (String) json.get("truthTableFileName");
      inputFileName = (String) json.get("inputFileName");
      outputFileName = (String) json.get("outputWeightsFileName");
      numPossibleInputSets = Math.toIntExact((long) json.get("numPossibleInputs"));
      
   } // public void parseConfigFile(String fileName)


/**
 * parses the json weights file using simple-json, assigns the arrays of weights
 */
   public void parseWeightsFile(String fileName) throws Exception
   {
      Object obj = null;
      JSONObject json = null;
      JSONArray weightskj;
      JSONArray weightsji;
      JSONArray weightsmk;
      int i;
      int j;
      int k;
      int m;

      try
      {
         obj = new JSONParser().parse(new FileReader(fileName));
      } catch (Exception e)
      {
         e.printStackTrace();
      }
      json = (JSONObject) obj;
      weightsmk = (JSONArray) json.get("weightsmk");
      weightskj = (JSONArray) json.get("weightskj");
      weightsji = (JSONArray) json.get("weightsji");

      if (weightskj.size()!=weights1.length || weightsji.size()!=weights2.length || weightsmk.size()!=weights0.length || 
      ((JSONArray) weightskj.get(0)).size()!=weights1[0].length || ((JSONArray) weightsji.get(0)).size()!=weights2[0].length 
      || ((JSONArray) weightsmk.get(0)).size()!=weights0[0].length) //check if dimensions of weights file are accurate
      {
         throw new Exception("Weights file does not match dimensionality in config file!");
      }
      
      for (k = 0; k < weights1.length; k++) 
      {
         for (j = 0; j < weights1[k].length; j++)
         {
            weights1[k][j] = (double) ((JSONArray) weightskj.get(k)).get(j);
         }
      }

      for (j = 0; j < weights2.length; j++) 
      {
         for (i = 0; i < weights2[j].length; i++)
         {
            weights2[j][i] = (double) ((JSONArray) weightsji.get(j)).get(i);
         }
      }

      for (m = 0; m < weights0.length; m++) 
      {
         for (k = 0; k < weights0[m].length; k++)
         {
            weights0[m][k] = (double) ((JSONArray) weightsmk.get(m)).get(k);
         }
      }
   } // public void parseWeightsFile(String fileName) throws Exception

/**
 * parses the truthTable json file using simple-json, assigns values to truthTable array
 */
   public void parseTruthTable(String fileName)
   {
      
      Object obj = null;
      JSONObject json = null;
      JSONArray truthTable;
      int j;
      int k;

      try
      {
         obj = new JSONParser().parse(new FileReader(fileName));
      } catch (Exception e)
      {
         e.printStackTrace();
      }

      json = (JSONObject) obj;
      truthTable = (JSONArray) json.get("truthTable");

      for (k = 0; k < truthTable.size(); k++) 
      {
         for (j = 0; j < ((JSONArray) truthTable.get(k)).size(); j++)
         {
            this.truthTable[k][j] =  (double) ((JSONArray) truthTable.get(k)).get(j);
         }
      }
   } // public void parseTruthTable(String fileName)

/**
 * parses the possibleInputs json file using simple-json, assigns possibleInputSets array
 */
   public void parseInputs(String fileName)
   {
      Object obj = null;
      JSONObject json = null;
      JSONArray inputs;
      int j;
      int k;

      try
      {
         obj = new JSONParser().parse(new FileReader(fileName));
      } catch (Exception e)
      {
         e.printStackTrace();
      }
      json = (JSONObject) obj;
      inputs = (JSONArray) json.get("possibleInputs");

      for (k = 0; k < inputs.size(); k++) 
      {
         possibleInputs[k] = parseInputFile((String) inputs.get(k));
      }
   } // public void parseInputs(String fileName)


   public double[] parseInputFile(String fileName)
   {
      double[] ret = new double[numInputs];
      int count = 0;
      BufferedReader reader = null;
      try
      {
         reader = new BufferedReader(new FileReader(fileName));
      }
      catch (Exception e)
      {
         e.printStackTrace();
      }

      String line = null;
      try {
         line = reader.readLine();
      } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      while (line != null)
      {
         ret[count] = Double.parseDouble(line);
         count++;
         try {
            line = reader.readLine();
         } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
         }
      }
      try {
         reader.close();
      } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      return ret;


   }

/**
 * saves the weights to a given file in json format
 * @param fileName file to be saved to
 */
   public void saveWeights(String fileName)
   {
      JSONObject output = new JSONObject();
      JSONArray weightsmk = new JSONArray();
      JSONArray weightskj = new JSONArray();
      JSONArray weightsji = new JSONArray();
      JSONArray tempweights = new JSONArray();
      PrintWriter pw = null;
      int k;
      int j;
      int i;
      int m;

      try
      {
         pw = new PrintWriter(fileName);
      }
      catch(Exception e)
      {
         e.printStackTrace();
         System.exit(0);
      }

      for (m = 0; m < weights0.length; m++)
      {
         tempweights = new JSONArray();
         for (k = 0; k < weights0[m].length; k++)
         {
            tempweights.add(weights0[m][k]);
         }
         weightsmk.add(tempweights);
      }

      for (k = 0; k < weights1.length; k++)
      {
         tempweights = new JSONArray();
         for (j = 0; j < weights1[k].length; j++)
         {
            tempweights.add(weights1[k][j]);
         }
         weightskj.add(tempweights);
      }

      
      for (j = 0; j < weights2.length; j++)
      {
         tempweights = new JSONArray();
         for (i = 0; i < weights2[j].length; i++)
         {
            tempweights.add(weights2[j][i]);
         }
         weightsji.add(tempweights);
      }

      output.put("weightsmk", weightsmk);
      output.put("weightskj", weightskj);
      output.put("weightsji", weightsji);

      pw.write(output.toJSONString());
      pw.flush();
      pw.close();
   } // public void saveWeights(String fileName)

/**
 * asks for user input to determine the mode of the network (run or train)
 */
   public void inputMode()
   {
      Scanner scan = new Scanner(System.in);
      int runMode = 0;
      System.out.print("Would you like to RUN(1) or TRAIN(2) the network: ");
      try 
      {
         runMode = scan.nextInt();
      } 
      catch (Exception e) 
      {
         e.printStackTrace();
      }

      while (runMode!=1 && runMode !=2)
      {
         System.out.println("Integer 1 or 2 not entered to determine network mode!");
         System.out.print("Would you like to RUN(1) or TRAIN(2) the network: ");
         try 
         {
            runMode = scan.nextInt();
         } 
         catch (Exception e) 
         {
            e.printStackTrace();
         }
      } //while (runMode!=1 && runMode !=2)
      scan.close();

      if (runMode==1)
      {
         System.out.println("Mode set to RUN!\n");
         runTrain = mode.RUN;
      }
      else
      {
         System.out.println("Mode set to TRAIN!\n");
         runTrain = mode.TRAIN;
      }
   } //public void inputMode()

/**
 * assigns all config values, sets the structure of the network and training hyperparameters
 */
   public void setConfigValues()
   {
      numInputs = 2;
      numHiddenNodes1 = 2;
      numHiddenNodes2 = 2;
      maxErrorThreshold = 0.001;
      lambda = 0.3;
      maxIterations = 100000;
      maxRandomWeight = 1.5;
      minRandomWeight = 0.1;
      numOutputs = 3;
      numIterations = 0;
   } //public void setConfigValues()


/**
 * echoes all config values assigned by setConfigValues()
 */
   public void echoConfigValues() 
   {
      System.out.println("Network Configuration:");
      System.out.println("\tNumber of Inputs: " + numInputs);
      System.out.println("\tNumber of Hidden Nodes in First Hidden Layer: " + numHiddenNodes1);
      System.out.println("\tNumber of Hidden Nodes in Second Hidden Layer: " + numHiddenNodes2);
      System.out.println("\tNumber of Outputs: " + numOutputs);

      if (runTrain.equals(mode.TRAIN))
      {
         System.out.println("\tError Threshold: " + maxErrorThreshold);
         System.out.println("\tLambda (Learning Rate): " + lambda);
         System.out.println("\tMaximum Number of Iterations: " + maxIterations);
      }

      if (!preLoadedWeights)
      {
         System.out.println("\tMaximum Initial Weight Value: " + maxRandomWeight);
         System.out.println("\tMinimum Initial Weight Value: " + minRandomWeight);
      }
      else
      {
         System.out.println("\tUsing Preloaded Weights From: <" + weightFileName + ">!");
      }

      System.out.println();

   } //public void echoConfigValues()

/**
 * allocates space for all instance variable arrays, e.g. weights, a, h, f, t, etc.
 */
   public void allocate()
   {
      weights0 = new double[numInputs][numHiddenNodes1];
      weights1 = new double[numHiddenNodes1][numHiddenNodes2];
      weights2 = new double[numHiddenNodes2][numOutputs];
      a = new double[numInputs];
      h1 = new double[numHiddenNodes1];
      h2 = new double[numHiddenNodes2];
      f = new double[numOutputs];
      t = new double[numOutputs];
      truthTable = new double[numPossibleInputSets][numOutputs];
      possibleInputs = new double[numPossibleInputSets][numInputs];

      if (runTrain.equals(mode.TRAIN))
      {
         thetak = new double[numHiddenNodes1];
         thetaj = new double[numHiddenNodes2];
         thetai = new double[numOutputs];
         psii = new double[numOutputs];
         omegai = new double[numOutputs];
         omegaj = new double[numHiddenNodes2];
         psij = new double[numHiddenNodes2];
         psik = new double[numHiddenNodes1];
         omegak = new double[numHiddenNodes1];
      } // if (runTrain.equals(mode.TRAIN))
      
   } //public void allocate()

/**
 * populates the truthtable and possible inputs with the training and testing data
 */
   public void populate()
   {
      parseTruthTable(truthTableFileName);
      parseInputs(inputFileName);

      if (!preLoadedWeights)
      {
         randomInitialization();
      }
      else
      {
         try
         {
            parseWeightsFile(weightFileName);
         }
         catch (Exception e)
         {
            e.printStackTrace();
            System.exit(0);
         }
      } // else
   } // public void populate()

/**
 * This function generates a random double between maxRandomWeight and minRandomWeight
 * Requires maxRandomWeight and minRandomWeight to both be doubles
 * @return double random weight
 */
   public double getRandWeight() 
   {
      return (Math.random() * (maxRandomWeight-minRandomWeight)) + minRandomWeight;
   }

/**
 * assigns random doubles ranging from minRandomWeight to maxRandomWeight to the weight arrays
 * weights0 and weights1. The method calls getRandWeight to generate the weights
 * Requires maxRandomWeight and minRandomWeight to both be doubles
 */
   public void randomInitialization() 
   {
      int k;
      int i;
      int j;
      int m;

      for (m = 0; m < weights0.length; m++)
      {
         for (k = 0; k < weights0[m].length; k++)
         {
            weights0[m][k] = getRandWeight();
         }
      }

      for (k = 0; k < weights1.length; k++) 
      {
         for (j = 0; j < weights1[k].length; j++)
         {
            weights1[k][j] = getRandWeight();
         }
      }

      for (j = 0; j < weights2.length; j++) 
      {
         for (i = 0; i < weights2[j].length; i++)
         {
            weights2[j][i] = getRandWeight();
         }
      }

   } //public void randomInitialization()

/**
 * return the sigmoid of the input double
 * @param x input into sigmoid function
 * @return double, return value of sigmoid
 */
   public double activationFunction(double x) 
   {
      return 1.0/(1.0+Math.exp(-x));
   }

/**
 * return the derivative of the sigmoid of the input double
 * @param x input into derivative of sigmoid function
 * @return double, return value of derivative of sigmoid
 */
   public double derivActivationFuncton(double x) 
   {
      double derivative = activationFunction(x);
      return derivative*(1.0-derivative);
   }

/**
 * runs the network with the given weight values and inputs, updates h, f, and stored theta, omega, psi values when training
 * @param input inputs to be placed into the input array (a)
 */
   public void runNetwork(double[] input)
   {
      int j;
      int i;
      int k;
      int m;
      a = input;

      for (k = 0; k < numHiddenNodes1; k++)
      {
         thetak[k] = 0.0;
         for (m = 0; m < numInputs; m++)
         {
            thetak[k] += a[m] * weights0[m][k];
         }
         h1[k] = activationFunction(thetak[k]);
      }

      for (j = 0; j < numHiddenNodes2; j++)
      {
         thetaj[j] = 0.0;
         for (k = 0; k < numHiddenNodes1; k++)
         {
            thetaj[j] += h1[k] * weights1[k][j];
         }
         h2[j] = activationFunction(thetaj[j]);
      }

      for (i = 0; i < numOutputs; i++)
      {
         thetai[i] = 0.0;
         for (j = 0; j < numHiddenNodes2; j++)
         {
            thetai[i] += h2[j] * weights2[j][i];
         }
         f[i] = activationFunction(thetai[i]);
         omegai[i] = t[i] - f[i];
         psii[i] = omegai[i] * derivActivationFuncton(thetai[i]);
      } // for (i = 0; i < numOutputs; i++)

   } // public void runNetwork(double[] input)

/**
 * executes the network with the given weight values and inputs, updates h, f, called when executing without training in mind
 * @param input inputs to be placed into the input array (a)
 */
   public void executeNetwork(double[] input)
   {
      int j;
      int i;
      int k;
      int m;
      double theta1;
      double theta2;
      double theta3;
      a = input;

      for (k = 0; k < numHiddenNodes1; k++)
      {
         theta1 = 0.0;
         for (m = 0; m < numInputs; m++)
         {
            theta1 += a[m] * weights0[m][k];
         }
         h1[k] = activationFunction(theta1);
      }

      for (j = 0; j < numHiddenNodes2; j++)
      {
         theta2 = 0.0;
         for (k = 0; k < numHiddenNodes1; k++)
         {
            theta2 += h1[k] * weights1[k][j];
         }
         h2[j] = activationFunction(theta2);
      }

      for (i = 0; i < numOutputs; i++)
      {
         theta3 = 0.0;
         for (j = 0; j < numHiddenNodes2; j++)
         {
            theta3 += h2[j] * weights2[j][i];
         }
         f[i] = activationFunction(theta3);
      } // for (i = 0; i < numOutputs; i++)


   } // public void executeNetwork(double[] input)

/**
 * trains the network using the truth table and possible inputs, updates weights according to design document to minimize error
 * updates t, weight arrays, and error instance variables
 */
   public void train() throws Exception
   {
      if (!runTrain.equals(mode.TRAIN))
      {
         throw new Exception("train() called when network mode not set to TRAIN!");
      }

      double totalError = Double.MAX_VALUE;
      int inputNum;
      int i;
      double[] currInput;

      while (numIterations<maxIterations && totalError > maxErrorThreshold) 
      {
         totalError = 0.0;

         for (inputNum = 0; inputNum < possibleInputs.length; inputNum++)
         {
            currInput = possibleInputs[inputNum];
            t = truthTable[inputNum];
            runNetwork(currInput);
            updateWeights();
            for (i = 0; i < f.length; i++)
            {
               totalError += 0.5 * ((t[i] - f[i]) * (t[i] - f[i]));
            }
         } // for (inputNum = 0; inputNum < possibleInputs.length; inputNum++)

         numIterations++;
      } // while(numIterations<maxIterations && totalError > maxErrorThreshold)

      error = totalError;
      System.out.println("Training results:");
      if (totalError <= maxErrorThreshold) 
      {
         System.out.println("\tTraining concluded: error (" + totalError + ") has reached error threshold (" + maxErrorThreshold + ")");
      }

      if (numIterations >= maxIterations ) 
      {
         System.out.println("\tNumber of iterations (" + numIterations + ") has reached maximum iterations allowed (" + maxIterations + ")");
      }
   } //public void train()

/**
 * updates the weights in accordance with the design document, calculating the partial derivatives
 * with respect to the error function, then changing the weights by their corresponding deltas
 */
   public void updateWeights() 
   {
      int j;
      int k;
      int i;
      int m;

      for (j = 0; j < numHiddenNodes2; j++)
      {
         omegaj[j] = 0.0;

         for (i = 0; i < numOutputs; i++)
         {
            omegaj[j] += psii[i] * weights2[j][i];
            weights2[j][i] += lambda * h2[j] * psii[i];
         }
         psij[j] = omegaj[j] * derivActivationFuncton(thetaj[j]);
      } // for (j = 0; j < numHiddenNodes2; j++)

      for (k = 0; k < numHiddenNodes1; k++)
      {
         omegak[k] = 0.0;
         for (j = 0; j < numHiddenNodes2; j++)
         {
            omegak[k] += psij[j]*weights1[k][j];
            weights1[k][j] += lambda * h1[k] * psij[j];
         }
         psik[k] = omegak[k] * derivActivationFuncton(thetak[k]);

         for (m = 0; m < numInputs; m++)
         {
            weights0[m][k] += lambda * a[m] * psik[k];
         }
      } // for (k = 0; k < numHiddenNodes1; k++)
   } //public void updateWeights

/**
 * runs the network over all possible inputs using current weights
 */
public void run() 
{
   double[] input;
   int i;

   for (i = 0; i<possibleInputs.length; i++) 
   {
      input = possibleInputs[i];
      executeNetwork(input);
   } //runs the network for all possible inputs

} //public void run()

/**
 * reports all pertinent information after running or training the network, including the number of iterations, the error, 
 * and all possible inputs and their corresponding outputs
 */
   public void report() 
   {
      int inputNum;
      int k;
      int i;

      if (runTrain.equals(mode.TRAIN))
      {

         System.out.println("\tNumber of iterations reached: " + numIterations);
         System.out.println("\tError reached: " + error);
         System.out.println("\tTraining time: " + trainingTime + "ms");
         System.out.println();  
      }

      System.out.println("Results: "); 
      for (inputNum = 0; inputNum < possibleInputs.length; inputNum++) 
      {
         executeNetwork(possibleInputs[inputNum]);
         for (k = 0; k < a.length; k++)
         {
            System.out.print("\t a" + k + " = " + String.format("%.4f", a[k])); //prints inputs to 4 decimals
         }
         for (i = 0; i < f.length; i++)
         {
            System.out.print("\t f" + i + " = " + String.format("%.4f", f[i])); //prints ouput to 4 decimals
            System.out.print("\t t" + i + " = " + String.format("%.4f", truthTable[inputNum][i])); //prints truth table to 4 decimals
         }
         System.out.print("\n");
      } // for (inputNum = 0; inputNum < possibleInputs.length; inputNum++)

   } //public void report()

/**
 * executes the network based on its predefined mode, either trains or runs the network using current weights
 */
   public void execute()
   {
      double startTime;
      double endTime;
      if (runTrain.equals(mode.TRAIN))
      {
         try 
         {
            startTime = System.currentTimeMillis();
            train();
            endTime = System.currentTimeMillis();
            trainingTime = endTime - startTime;
         } 
         catch (Exception e) 
         {
            e.printStackTrace();
         }
      } //if (runTrain.equals(mode.TRAIN))
      else
      {
         run();
      }
   } //public void execute()

/**
 * main method to test the network, trains and reports all pertinent information
 */
   public static void main(String[] args)
   {
      Network net = null;
      if (args.length==0)
      {
         net = new Network("configFile.json");
      }
      else
      {
         net = new Network(args[0]);
      }
      net.execute();
      net.report();
      if (net.runTrain.equals(mode.TRAIN) && net.outputFileName!=null)
      {
         net.saveWeights(net.outputFileName);
      }
   } // public static void main(String[] args)

} //public class Network
