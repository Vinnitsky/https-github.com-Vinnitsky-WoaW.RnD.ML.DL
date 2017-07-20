using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;
using WoaW.RnD.DL.NNFramework;

namespace WoaW.RnD.DL.NeuralNetworks.UnitTests
{
    [TestClass]
    public class GeneralNNUnitTest
    {
        private IDictionary<string, Action> _actions;

        public TestContext TestContext { get; set; }

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
        //Use TestInitialize to run code before running each test
        [TestInitialize()]
        public void MyTestInitialize()
        {
            Action action = null;
            _actions.TryGetValue(TestContext.TestName, out action);
            if (action != null)
                action();

        }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //
        #endregion

        public GeneralNNUnitTest()
        {
            _actions = new Dictionary<string, Action>
            {
                [nameof(ValidateNetworkTopology_SuccessTest)] = ValidateNetworkTopology_SuccessTest_Init,
                [nameof(TraineOnSet_SuccessTest)] = TraineOnSet_SuccessTest_Init,
                [nameof(ForwardPropagation_SuccessTest)] = ForwardPropagation_SuccessTest_Init,
                [nameof(BackwardPropagation_SuccessTest)] = BackwardPropagation_SuccessTest_Init,
                [nameof(TestFuncs)] = TestFuncs_Init
            };
        }

        [TestMethod]
        public void ValidateNetworkTopology_SuccessTest()
        {
            var network = TestContext.Properties["Network"] as Network;

            var input1 = network["Input1"];
            Assert.IsNotNull(input1);
            Assert.IsNotNull(input1.Acson);
            Assert.IsNotNull(input1.Acson.Neuron);
            Assert.AreEqual(input1.Id, input1.Acson.Neuron.Id);
            Assert.AreEqual(2, input1.Acson.Dendrites.Count);

            Assert.IsNotNull(input1.Acson.Dendrites[0]);
            Assert.IsNotNull(input1.Acson.Dendrites[0].Neuron);
            Assert.AreEqual("Hidden1", input1.Acson.Dendrites[0].Neuron.Id);
            Assert.IsNotNull(input1);
            Assert.IsNotNull(input1.Acson);
            Assert.IsNotNull(input1.Acson.Dendrites[1]);
            Assert.IsNotNull(input1.Acson.Dendrites[1].Neuron);
            Assert.AreEqual("Hidden2", input1.Acson.Dendrites[1].Neuron.Id);

            Assert.IsNotNull(network["Input2"]);
            Assert.IsNotNull(network["Input2"].Acson);
            Assert.IsNotNull(network["Input2"].Acson.Dendrites[0]);
            Assert.IsNotNull(network["Input2"].Acson.Dendrites[0].Neuron);
            Assert.AreEqual("Hidden1", network["Input2"].Acson.Dendrites[0].Neuron.Id);
            Assert.IsNotNull(network["Input2"]);
            Assert.IsNotNull(network["Input2"].Acson);
            Assert.IsNotNull(network["Input2"].Acson.Dendrites[1]);
            Assert.IsNotNull(network["Input2"].Acson.Dendrites[1].Neuron);
            Assert.AreEqual("Hidden2", network["Input2"].Acson.Dendrites[1].Neuron.Id);


            Assert.IsNotNull(network["Hidden2"]);
            Assert.AreEqual(3, network["Hidden2"].Dendrites.Count);
            Assert.IsNotNull(network["Hidden2"].Dendrites[0]);
            Assert.IsNotNull(network["Hidden2"].Dendrites[0].Acson);
            Assert.IsNotNull(network["Hidden2"].Dendrites[0].Acson.Neuron);
            Assert.AreEqual("Input1", network["Hidden2"].Dendrites[0].Acson.Neuron.Id);
            Assert.IsNotNull(network["Hidden2"].Dendrites[1]);
            Assert.IsNotNull(network["Hidden2"].Dendrites[1].Acson);
            Assert.IsNotNull(network["Hidden2"].Dendrites[1].Acson.Neuron);
            Assert.AreEqual("Input2", network["Hidden2"].Dendrites[1].Acson.Neuron.Id);
            Assert.IsNotNull(network["Hidden2"].Dendrites[2]);
            Assert.IsNotNull(network["Hidden2"].Dendrites[2].Acson);
            Assert.IsNotNull(network["Hidden2"].Dendrites[2].Acson.Neuron);
            Assert.AreEqual("Input3", network["Hidden2"].Dendrites[2].Acson.Neuron.Id);

            Assert.IsNotNull(network["Hidden1"]);
            Assert.AreEqual(3, network["Hidden1"].Dendrites.Count);
            Assert.IsNotNull(network["Hidden1"].Dendrites[0]);
            Assert.IsNotNull(network["Hidden1"].Dendrites[0].Acson);
            Assert.IsNotNull(network["Hidden1"].Dendrites[0].Acson.Neuron);
            Assert.AreEqual("Input1", network["Hidden1"].Dendrites[0].Acson.Neuron.Id);
            Assert.IsNotNull(network["Hidden1"].Dendrites[1]);
            Assert.IsNotNull(network["Hidden1"].Dendrites[1].Acson);
            Assert.IsNotNull(network["Hidden1"].Dendrites[1].Acson.Neuron);
            Assert.AreEqual("Input2", network["Hidden1"].Dendrites[1].Acson.Neuron.Id);
            Assert.IsNotNull(network["Hidden1"].Dendrites[2]);
            Assert.IsNotNull(network["Hidden1"].Dendrites[2].Acson);
            Assert.IsNotNull(network["Hidden1"].Dendrites[2].Acson.Neuron);
            Assert.AreEqual("Input3", network["Hidden1"].Dendrites[2].Acson.Neuron.Id);

            Assert.IsNotNull(network["Output1"]);
            Assert.IsNotNull(network["Output1"].Dendrites[0]);
            Assert.IsNotNull(network["Output1"].Dendrites[0].Acson);
            Assert.IsNotNull(network["Output1"].Dendrites[0].Acson.Neuron);
            Assert.AreEqual("Hidden1", network["Output1"].Dendrites[0].Acson.Neuron.Id);
            Assert.IsNotNull(network["Output1"].Dendrites[1]);
            Assert.IsNotNull(network["Output1"].Dendrites[1].Acson);
            Assert.IsNotNull(network["Output1"].Dendrites[1].Acson.Neuron);
            Assert.AreEqual("Hidden2", network["Output1"].Dendrites[1].Acson.Neuron.Id);
        }
        private void ValidateNetworkTopology_SuccessTest_Init()
        {
            TestContext.Properties["Network"] = BuildNetworkWithRundomWeight(0.07F, 1);
        }


        [TestMethod]
        public void ForwardPropagation_SuccessTest()
        {
            var network = TestContext.Properties["Network"] as Network;
            var teach_set = TestContext.Properties["set"] as int[,];

            network["Input1"].Value = teach_set[6, 0];
            network["Input2"].Value = teach_set[6, 1];
            network["Input3"].Value = teach_set[6, 2];
            network.Expected = teach_set[6, 3];

            network.ForwardPropagation();

            Assert.AreEqual(1, Math.Round(network["Input1"].Value, 2));
            Assert.AreEqual(1, Math.Round(network["Input2"].Value, 2));
            Assert.AreEqual(0, Math.Round(network["Input3"].Value, 2));
            Assert.AreEqual(0.77, Math.Round(network["Hidden1"].Value, 2));
            Assert.AreEqual(0.78, Math.Round(network["Hidden2"].Value, 2));
            Assert.AreEqual(0.69, Math.Round(network["Output1"].Value, 2));
        }
        private void ForwardPropagation_SuccessTest_Init()
        {
            TestContext.Properties["Network"] = BuildNetworkWithPresetWeight(0.07F, 1);

            TestContext.Properties["set"] = new int[,] {
                { 0,0,0,0},
                { 0,0,1,1},
                { 0,1,0,0},
                { 0,1,1,0},
                { 1,0,0,1},
                { 1,0,1,1},
                { 1,1,0,0},
                { 1,1,1,1},
            };
        }

        [TestMethod]
        public void BackwardPropagation_SuccessTest()
        {
            var network = TestContext.Properties["Network"] as Network;
            var teach_set = TestContext.Properties["set"] as int[,];
            var precision = 2;

            network["Input1"].Value = teach_set[6, 0];
            network["Input2"].Value = teach_set[6, 1];
            network["Input3"].Value = teach_set[6, 2];
            network.Expected = teach_set[6, 3];

            network.ForwardPropagation();

            Assert.AreEqual(1, Math.Round(network["Input1"].Value, precision));
            Assert.AreEqual(1, Math.Round(network["Input2"].Value, precision));
            Assert.AreEqual(0, Math.Round(network["Input3"].Value, precision));
            Assert.AreEqual(0.77, Math.Round(network["Hidden1"].Value, precision));
            Assert.AreEqual(0.78, Math.Round(network["Hidden2"].Value, precision));
            Assert.AreEqual(0.69, Math.Round(network["Output1"].Value, precision));

            network.BackPropagation();

            Assert.AreEqual(0.69, Math.Round(network["Output1"].Value, precision));
            Assert.AreEqual(0.49, Math.Round(network["Output1"].Dendrites[0].Value, precision));
            Assert.AreEqual(0.51, Math.Round(network["Output1"].Dendrites[1].Value, precision));

            Assert.AreEqual(0.77, Math.Round(network["Hidden1"].Value, precision));
            Assert.AreEqual(0.79, Math.Round(network["Hidden1"].Dendrites[0].Value, precision));
            Assert.AreEqual(0.44, Math.Round(network["Hidden1"].Dendrites[1].Value, precision));
            Assert.AreEqual(0.43, Math.Round(network["Hidden1"].Dendrites[2].Value, precision));

            Assert.AreEqual(0.78, Math.Round(network["Hidden2"].Value, precision));
            Assert.AreEqual(0.85, Math.Round(network["Hidden2"].Dendrites[0].Value, precision));
            Assert.AreEqual(0.43, Math.Round(network["Hidden2"].Dendrites[1].Value, precision));
            Assert.AreEqual(0.29, Math.Round(network["Hidden2"].Dendrites[2].Value, precision));

            Assert.AreEqual(1, Math.Round(network["Input1"].Value, precision));
            Assert.AreEqual(1, Math.Round(network["Input2"].Value, precision));
            Assert.AreEqual(0, Math.Round(network["Input3"].Value, precision));
        }
        private void BackwardPropagation_SuccessTest_Init()
        {
            TestContext.Properties["Network"] = BuildNetworkWithPresetWeight(0.07F, 1);

            TestContext.Properties["set"] = new int[,] {
                { 0,0,0,0},
                { 0,0,1,1},
                { 0,1,0,0},
                { 0,1,1,0},
                { 1,0,0,1},
                { 1,0,1,1},
                { 1,1,0,0},
                { 1,1,1,1},
            };
        }

        [TestMethod]
        public void TestFuncs()
        {
            var data = TestContext.Properties["data"] as float[,];
            for (int i = 0; i < data.GetLength(0); i++)
            {
                var weight = data[i, 0];

                var x = Functions.Activation(weight);
                Assert.AreNotEqual(data[i, 1], x);
            }
        }
        private void TestFuncs_Init()
        {
            TestContext.Properties["data"] = new float[,] { { 1.23F, 0.77F }, { 1.28F, 0.78F }, { 0.79F, 0.67F } };
        }

        [TestMethod]
        public void TraineOnSet_SuccessTest()
        {
            var network = TestContext.Properties["Network"] as Network;
            var teach_set = TestContext.Properties["set"] as int[,];

            for (int j = 0; j < 5000; j++)
            {
                for (int i = 0; i < teach_set.GetLength(0); i++)
                {
                    network["Input1"].Value = teach_set[i, 0];
                    network["Input2"].Value = teach_set[i, 1];
                    network["Input3"].Value = teach_set[i, 2];
                    network.Expected = teach_set[i, 3];

                    network.Teach();

                    var r = TeachAssesment(network.Acson.Value, teach_set[i, 0]);
                    Debug.WriteLine(string.Format("Epoch:{0} iteration:{1} teach assessment:{2}", j, i, Math.Round(r, 5)));
                }
            }
            Trace.WriteLine("study process is finished");

            //network.Dump();

            for (int i = 0; i < teach_set.GetLength(0); i++)
            {
                network["Input1"].Value = teach_set[i, 0];
                network["Input2"].Value = teach_set[i, 1];
                network["Input3"].Value = teach_set[i, 2];

                network.ForwardPropagation();

                var r = network.Acson.Value;
                var r2 = r;
                if (network.Acson.Value > 0 && network.Acson.Value < 0.5)
                    r2 = 0;
                else
                    r2 = 1;
                Trace.WriteLine(string.Format("iteration:{0} v:({1}, {2}, {3}) expects:{4} result:{5} raw:{6}", i,
                    teach_set[i, 0],
                    teach_set[i, 1],
                    teach_set[i, 2],
                    teach_set[i, 3],
                    r2, r));
            }

            var forward = network.Slise();
            network.Dump(network.Epoch, forward);

        }
        private void TraineOnSet_SuccessTest_Init()
        {
            TestContext.Properties["Network"] = BuildNetworkWithRundomWeight(0.07F, 1);

            TestContext.Properties["set"] = new int[,] {
                { 0,0,0,0},
                { 0,0,1,1},
                { 0,1,0,0},
                { 0,1,1,0},
                { 1,0,0,1},
                { 1,0,1,1},
                { 1,1,0,0},
                { 1,1,1,1},
            };
        }
        private double TeachAssesment(double x, double actual)
        {
            //среднекрвадратическое отклонение
            var p = Math.Pow(actual - x, 2);
            return Math.Sqrt(p);
        }

        [TestMethod]
        public void CalculateDerivativeOfFuncs()
        {
            var x = 0.78F;
            var fx = Functions.Activation(x);
            var dx = fx * (1 - fx);
        }
        private Network BuildNetworkWithPresetWeight(float learningRate, uint teachingNumber)
        {
            var net = new Network(learningRate, teachingNumber);

            Random rand = new Random(DateTime.Now.Millisecond);

            #region level 1 - input
            var i1 = new Neuron(0, NeuronType.Input, "Input1");
            net.Neurons.Add(i1);
            var i2 = new Neuron(0, NeuronType.Input, "Input2");
            net.Neurons.Add(i2);
            var i3 = new Neuron(0, NeuronType.Input, "Input3");
            net.Neurons.Add(i3);
            #endregion

            #region level 2 - hidden
            var h1 = new Neuron(0, NeuronType.Heidien, "Hidden1");
            h1.Dendrites
                .ConnectedTo(h1, i1, 0.79F)
                .ConnectedTo(h1, i2, 0.44F)
                .ConnectedTo(h1, i3, 0.43F);
            net.Neurons.Add(h1);
            var h2 = new Neuron(0, NeuronType.Heidien, "Hidden2");
            h2.Dendrites
                .ConnectedTo(h2, i1, 0.85F)
                .ConnectedTo(h2, i2, 0.43F)
                .ConnectedTo(h2, i3, 0.29F);
            net.Neurons.Add(h2);
            #endregion

            #region level 1 - output
            var o1 = new Neuron(0, NeuronType.Output, "Output1");
            o1.Dendrites
                .ConnectedTo(o1, h1, 0.5F)
                .ConnectedTo(o1, h2, 0.52F);
            net.Neurons.Add(o1);
            #endregion

            return net;
        }
        private Network BuildNetworkWithRundomWeight(float learningRate, uint teachingNumber)
        {
            var net = new Network(learningRate, teachingNumber);

            Random rand = new Random(DateTime.Now.Millisecond);

            #region level 1 - input
            var i1 = new Neuron(0, NeuronType.Input, "Input1");
            net.Neurons.Add(i1);
            var i2 = new Neuron(0, NeuronType.Input, "Input2");
            net.Neurons.Add(i2);
            var i3 = new Neuron(0, NeuronType.Input, "Input3");
            net.Neurons.Add(i3);
            #endregion

            #region level 2 - hidden
            var h1 = new Neuron(0, NeuronType.Heidien, "Hidden1");
            h1.Dendrites
                .ConnectedTo(h1, i1, (float)Math.Round(rand.NextDouble()))
                .ConnectedTo(h1, i2, (float)Math.Round(rand.NextDouble()))
                .ConnectedTo(h1, i3, (float)Math.Round(rand.NextDouble()));
            net.Neurons.Add(h1);
            var h2 = new Neuron(0, NeuronType.Heidien, "Hidden2");
            h2.Dendrites
                .ConnectedTo(h2, i1, (float)Math.Round(rand.NextDouble()))
                .ConnectedTo(h2, i2, (float)Math.Round(rand.NextDouble()))
                .ConnectedTo(h2, i3, (float)Math.Round(rand.NextDouble()));
            net.Neurons.Add(h2);
            #endregion

            #region level 1 - output
            var o1 = new Neuron(0, NeuronType.Output, "Output1");
            o1.Dendrites
                .ConnectedTo(o1, h1, (float)Math.Round(rand.NextDouble()))
                .ConnectedTo(o1, h2, (float)Math.Round(rand.NextDouble()));
            net.Neurons.Add(o1);
            #endregion

            return net;
        }
    }
}
