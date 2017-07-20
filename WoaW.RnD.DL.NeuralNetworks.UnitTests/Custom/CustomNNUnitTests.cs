using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

namespace WoaW.RnD.ML.NN.Tests
{
    /// <summary>
    /// Summary description for UnitTest1
    /// </summary>
    [TestClass]
    public class CustomNNUnitTests
    {
        public CustomNNUnitTests()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Additional test attributes
        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //
        #endregion

        [TestMethod]
        public void TestMethod1()
        {
            List<List<float>> train = new List<List<float>> {
                new List<float>{0, 0, 0},
                new List<float>{0, 0, 1},
                new List<float>{0, 1, 0},
                new List<float>{0, 1, 1},
                new List<float>{1, 0, 0},
                new List<float>{1, 0, 1},
                new List<float>{1, 1, 0},
                new List<float>{1, 1, 1},
            };
            List<float> correct_predictions = new List<float> { 0, 1, 0, 0, 1, 1, 0, 0, };
            var learningRate = 0.08F;
            var epoch = 5000;

            var exit = false;
            var network = new CustomNN(learningRate);
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < train.Count; j++)
                {
                    var input_stat = train[j];
                    var correct_predict = correct_predictions[j];
                    network.train(input_stat, correct_predict);
                    correct_predictions.Add(correct_predict);

                    var predict = network.predict(train[j]);
                    var train_loss = network.MSE(predict, correct_predictions);
                    System.Diagnostics.Debug.WriteLine(string.Format(" Progress: {0}% , Training loss: {1}",
                        100 * i / epoch, Math.Round(train_loss, 5)));

                    if (Math.Round(train_loss,4) == 0)
                    {
                        exit = true;
                        break;
                    }
                }
                if (exit == true)
                    break;
            }

            for (int i = 0; i < train.Count; i++)
            {
                var predict = network.predict(train[i]);

                System.Diagnostics.Debug.WriteLine(string.Format(" expected: {0} , actual: {1}",
                    correct_predictions[i], predict[0]));
            }
        }
    }
}
