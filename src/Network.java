/**
 * Pranav Varmaraja
 * 2/25/2022
 * 
 * This class creates, trains, and runs a basic three activation layer perceptron. The network takes any number of inputs, 
 * passes them through 1 activation layer consisting of any number of nodes, and outputs any number of values. The network
 * is thus an A-B-C network, with any number of inputs, hidden nodes, and outputs.
 * Training is performed via gradient descent to fit the network to training data.
 * 
 * Methods contained in file: setConfigValues(), echoConfigValues(), 
 * allocate(), populate(), randomInitialization(), getRandWeight(), activationFunction(x), 
 * derivActivationFunction(x), runNetwork(input), run(), train(), report(), updateWeight()
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
   String mode;

   double[][] possibleInputs;
   double[] truthTable;

   double[] thetaj;
   double[] thetai;
   double[] omegaj;
   double[] omegai;
   double[] psij;
   double[] psii;

/**
 * constructor for the Network, sets the config values, echoes then, allocates and populates all instance variable arrays
 */
   public Network() 
   {
      setConfigValues();
      echoConfigValues();
      allocate();
      populate();
   }

/**
 * assigns all config values, sets the structure of the network and training hyperparameters
 */
   public void setConfigValues()
   {
      mode = "TRAIN";
      numInputs = 2;
      numHiddenNodes = 2;
      maxErrorThreshold = 0.001;
      lambda = 0.3;
      maxIterations = 100000;
      maxRandomWeight = 1.5;
      minRandomWeight = 0.1;
      numOutputs = 1;
      numPossibleInputSets = 4;
      numIterations = 0;
   } //public void setConfigValues()


/**
 * echoes all config values assigned by setConfigValues()
 */
   public void echoConfigValues() 
   {
      System.out.println("Number of Inputs: " + numInputs);
      System.out.println("Number of Hidden Nodes: " + numHiddenNodes);
      System.out.println("Number of Outputs: " + numOutputs);
      System.out.println("Error Threshold: " + maxErrorThreshold);
      System.out.println("Lambda (Learning Rate): " + lambda);
      System.out.println("Maximum Number of Iterations: " + maxIterations);
      System.out.println("Maximum Initial Weight Value: " + maxRandomWeight);
      System.out.println("Minimum Initial Weight Value: " + minRandomWeight);

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

      if (mode.equals("TRAIN"))
      {
         thetaj = new double[numHiddenNodes];
         thetai = new double[numOutputs];
         omegaj = new double[numHiddenNodes];
         omegai = new double[numOutputs];
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
      truthTable = new double[] {0.0,1.0,1.0,0.0};
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
 * runs the network with the given weight values and inputs, updates h, f, and stored theta values
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
         thetaj[j] = theta;
         h[j] = activationFunction(theta);
      }

      for (i = 0; i < f.length; i++)
      {
         theta = 0.0;
         for (j = 0; j < h.length; j++)
         {
            theta += h[j]*weights1[j][i];
         }
         thetai[i] = theta;
         f[i] = activationFunction(theta);
      }

   } //public void runNetwork()

/**
 * trains the network using the truth table and possible inputs, updates weights according to design document to minimize error
 * updates t, weight arrays, and error instance variables
 */
   public void train() throws Exception
   {
      if (!mode.equals("TRAIN"))
      {
         throw new Exception("train() called when network mode not set to \"TRAIN\"!");
      }

      double totalError = Double.MAX_VALUE;
      int i;
      double[] currInput;

      while (numIterations<maxIterations && totalError > maxErrorThreshold) 
      {
         totalError = 0.0;

         for (i = 0; i < possibleInputs.length; i++)
         {
            currInput = possibleInputs[i];
            t[0] = truthTable[i];
            runNetwork(currInput);
            updateWeights();
            totalError += 0.5*((truthTable[i]-f[0])*(truthTable[i]-f[0]));
         }

         numIterations++;
      } // while(numIterations<maxIterations && totalError > maxErrorThreshold)

      error = totalError;

      if (totalError <= maxErrorThreshold) 
      {
         System.out.println("Training concluded: error (" + totalError + ") has reached error threshold (" + maxErrorThreshold + ")");
      }

      if (numIterations >= maxIterations ) 
      {
         System.out.println("Number of iterations (" + numIterations + ") has reached maximum iterations allowed (" + maxIterations + ")");
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
      double lowerOmega = t[0] - f[0];
      double psi0 = lowerOmega*derivActivationFuncton(thetai[0]); //assigns psi0 in accordance with design document

      for (j = 0; j<omegaj.length; j++) 
      {
         omegaj[j] = psi0*weights1[j][0];
         psij[j] = omegaj[j]*derivActivationFuncton(thetaj[j]);
      }

      for (k =0; k<weights0.length; k++) 
      {
         for (j = 0; j<weights0[k].length; j++) 
         {
            weights0[k][j] += -lambda*-a[k]*psij[j];
         }
      }

      for (j = 0; j<weights1.length; j++) 
      {
         for (i = 0; i<weights1[j].length; i++) 
         {
            weights1[j][i] += -lambda*-h[j]*psi0;
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
 * reports all pertinent information after training the network, including the number of iterations, the error, 
 * and all possible inputs and their corresponding outputs
 */
   public void report() 
   {
      int i;
      int j;

      System.out.println("Number of iterations reached: " + numIterations);
      System.out.println("Error reached: " + error);
      System.out.println("Truth Table: ");

      for (i = 0; i < possibleInputs.length; i++) 
      {
         runNetwork(possibleInputs[i]);
         for (j = 0; j < a.length; j++)
         {
            System.out.print("\t a" + j + " = " + a[j]); //prints inputs
         }
         System.out.print("\tf = " + f[0]); //prints output
         System.out.print("\tt = " + truthTable[i] + "\n"); //prints truth table  
      }
   } //public void report()

/**
 * main method to test the network, trains and reports all pertinent information
 */
   public static void main(String[] args) throws Exception 
   {
      Network net = new Network();
      net.train();
      net.report();
   }

} //public class Network
