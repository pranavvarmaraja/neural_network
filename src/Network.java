import java.util.Scanner;

/**
 * Pranav Varmaraja
 * 3/11/2022
 * 
 * This class creates, trains, and runs a basic three activation layer perceptron. The network takes any number of inputs, 
 * passes them through 1 activation layer consisting of any number of nodes, and outputs any number of values. The network
 * is thus an A-B-C network, with any number of inputs, hidden nodes, and outputs.
 * Training is performed via gradient descent to fit the network to training data.
 * 
 * Methods contained in file: inputMode(), setConfigValues(), echoConfigValues(), 
 * allocate(), populate(), randomInitialization(), getRandWeight(), activationFunction(x), 
 * derivActivationFunction(x), runNetwork(input), run(), train(), execute(), report(), updateWeight()
 * 
 */
public class Network 
{

   double[][] weights0;
   double[][] weights1;
   double[] a;
   double[] h;
   double[] f;
   double[] t;

   enum mode {RUN, TRAIN};
   mode runTrain;

   int numInputs;
   int numOutputs;
   int numHiddenNodes;
   double maxErrorThreshold;
   double lambda;
   int maxIterations;
   double maxRandomWeight;
   double minRandomWeight;
   int numPossibleInputSets;
   int numIterations;
   double error;

   double[][] possibleInputs;
   double[][] truthTable;

   double[] thetaj;
   double[] thetai;
   double[] omegaj;
   double[] psij;
   double[] psii;

/**
 * constructor for the Network, sets the config values, echoes then, allocates and populates all instance variable arrays
 */
   public Network() 
   {
      inputMode();
      setConfigValues();
      echoConfigValues();
      allocate();
      populate();
   }

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
      numHiddenNodes = 5;
      maxErrorThreshold = 0.001;
      lambda = 0.3;
      maxIterations = 100000;
      maxRandomWeight = 1.5;
      minRandomWeight = 0.1;
      numOutputs = 3;
      numPossibleInputSets = 4;
      numIterations = 0;
   } //public void setConfigValues()


/**
 * echoes all config values assigned by setConfigValues()
 */
   public void echoConfigValues() 
   {
      System.out.println("Network Configuration:");
      System.out.println("\tNumber of Inputs: " + numInputs);
      System.out.println("\tNumber of Hidden Nodes: " + numHiddenNodes);
      System.out.println("\tNumber of Outputs: " + numOutputs);

      if (runTrain.equals(mode.TRAIN))
      {
         System.out.println("\tError Threshold: " + maxErrorThreshold);
         System.out.println("\tLambda (Learning Rate): " + lambda);
         System.out.println("\tMaximum Number of Iterations: " + maxIterations);
      }

      System.out.println("\tMaximum Initial Weight Value: " + maxRandomWeight);
      System.out.println("\tMinimum Initial Weight Value: " + minRandomWeight);
      System.out.println();

   } //public void echoConfigValues()

/**
 * allocates space for all instance variable arrays, e.g. weights, a, h, f, t, etc.
 */
   public void allocate()
   {
      weights0 = new double[numInputs][numHiddenNodes];
      weights1 = new double[numHiddenNodes][numOutputs];
      a = new double[numInputs];
      h = new double[numHiddenNodes];
      f = new double[numOutputs];
      t = new double[numOutputs];
      possibleInputs = new double[numPossibleInputSets][numInputs];

      if (runTrain.equals(mode.TRAIN))
      {
         thetaj = new double[numHiddenNodes];
         thetai = new double[numOutputs];
         omegaj = new double[numHiddenNodes];
         psij = new double[numHiddenNodes];
         psii = new double[numOutputs];
      }
      
   } //public void allocate()

/**
 * populates the truthtable and possible inputs with the training and testing data
 */
   public void populate() 
   {
      possibleInputs = new double[][] { new double[] {0.0, 0.0}, new double[] {0.0, 1.0},  new double[] {1.0, 0.0}, new double[] {1.0, 1.0}};
      truthTable = new double[][] {new double[] {0.0, 0.0, 0.0}, new double[] {0.0, 1.0, 1.0}, new double[] {0.0, 1.0, 1.0}, new double[] {1.0, 1.0, 0.0}};
      randomInitialization();
   }

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

      for (k = 0; k < weights0.length; k++) 
      {
         for (j = 0; j < weights0[k].length; j++)
         {
            weights0[k][j] = getRandWeight();
         }
      }

      for (j = 0; j < weights1.length; j++) 
      {
         for (i = 0; i < weights1[j].length; i++)
         {
            weights1[j][i] = getRandWeight();
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
 * runs the network with the given weight values and inputs, updates h, f, and stored theta values (if training)
 * @param input inputs to be placed into the input array (a)
 */
   public void runNetwork(double[] input)
   {
      int k;
      int i;
      int j;
      double theta;

      a = input;

      for (j = 0; j < h.length; j++) 
      {
         theta = 0.0;
         for (k = 0; k < a.length; k++) 
         {
            theta += a[k] * weights0[k][j];
         }
         if (runTrain.equals(mode.TRAIN))
         {
            thetaj[j] = theta;
         }
         h[j] = activationFunction(theta);
      } //for (j = 0; j < h.length; j++)

      for (i = 0; i < f.length; i++)
      {
         theta = 0.0;
         for (j = 0; j < h.length; j++)
         {
            theta += h[j]*weights1[j][i];
         }
         if (runTrain.equals(mode.TRAIN))
         {
            thetai[i] = theta;
         }
         f[i] = activationFunction(theta);
      } //for (i = 0; i < f.length; i++)

   } //public void runNetwork()

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
      int k;
      int j;
      int i;

      for ( i = 0; i<numOutputs; i++) 
      {
         psii[i] = (t[i]-f[i])*derivActivationFuncton(thetai[i]);
      }

      for (j = 0; j<omegaj.length; j++) 
      {
         omegaj[j] = 0.0;
         for (i = 0; i< psii.length; i++)
         {
            omegaj[j] += psii[i] * weights1[j][i];
         }

         psij[j] = omegaj[j]*derivActivationFuncton(thetaj[j]);
      }

      for (k =0; k<weights0.length; k++) 
      {
         for (j = 0; j<weights0[k].length; j++) 
         {
            weights0[k][j] += lambda*a[k]*psij[j];
         }
      }

      for (j = 0; j<weights1.length; j++) 
      {
         for (i = 0; i<weights1[j].length; i++) 
         {
            weights1[j][i] += lambda*h[j]*psii[i];
         }
      }

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
      runNetwork(input);
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
         System.out.println();  
      }

      System.out.println("Results: "); 
      for (inputNum = 0; inputNum < possibleInputs.length; inputNum++) 
      {
         runNetwork(possibleInputs[inputNum]);
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
      if (runTrain.equals(mode.TRAIN))
      {
         try 
         {
            train();
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
   public static void main(String[] args) throws Exception 
   {
      Network net = new Network();
      net.execute();
      net.report();
   }

} //public class Network
