using System;
using System.Collections.Generic;

namespace WoaW.RnD.ML.NN.Tests
{
    class CustomNN
    {
        private float _learningRate;
        public List<List<float>> weights_0_1 = new List<List<float>>();
        public List<List<float>> weights_1_2 = new List<List<float>>();
        private int _precision = 5;

        public CustomNN(float learningRate)
        {
            _learningRate = learningRate;

            Random rand = new Random(DateTime.Now.Millisecond);

            weights_0_1.Add(new List<float> { (float)Math.Round(rand.NextDouble(), _precision), (float)Math.Round(rand.NextDouble(), _precision), (float)Math.Round(rand.NextDouble(), _precision) });
            weights_0_1.Add(new List<float> { (float)Math.Round(rand.NextDouble(), _precision), (float)Math.Round(rand.NextDouble(), _precision), (float)Math.Round(rand.NextDouble(), _precision) });

            weights_1_2.Add(new List<float> { (float)Math.Round(rand.NextDouble(), _precision), (float)Math.Round(rand.NextDouble(), _precision) });
        }

        internal void train(List<float> inputs, float expected_predict)
        {
            //forward propagation
            var outputs_1 = sigmoid_mapper(weights_0_1, inputs);
            var outputs_2 = sigmoid_mapper(weights_1_2, outputs_1);

            //backward propagation
            List<float> actual_predict2 = outputs_2;
            var error_layer_2_list = new List<float>();
            for (int i = 0; i < actual_predict2.Count; i++)
            {
                var error_layer = actual_predict2[i] - expected_predict;
                error_layer_2_list.Add(error_layer);
                for (int j = 0; j != weights_1_2[i].Count; j++)
                {
                    var gradient_layer = actual_predict2[i] * (1 - actual_predict2[i]);// derivative
                    var weights_delta_layer = error_layer * gradient_layer;
                    weights_1_2[i][j] = weights_1_2[i][j] - (weights_delta_layer * actual_predict2[i]) * _learningRate;
                }
            }

            var actual_predict1 = outputs_1;
            var error_layer_1_list = new List<float>();
            for (int i = 0; i < actual_predict1.Count; i++)
            {
                var error_layer = actual_predict1[i] - expected_predict;
                error_layer_1_list.Add(error_layer);
                for (int j = 0; j != weights_0_1[i].Count; j++)
                {
                    var gradient_layer = actual_predict1[i] * (1 - actual_predict1[i]);// derivative
                    var weights_delta_layer = error_layer * gradient_layer;
                    //!!!!
                    weights_0_1[i][j] = weights_0_1[i][j] - (weights_delta_layer * actual_predict1[i]) * _learningRate;
                }
            }

        }
        internal List<float> predict(List<float> inputs)
        {
            var outputs_1 = sigmoid_mapper(weights_0_1, inputs);

            var outputs_2 = sigmoid_mapper(weights_1_2, outputs_1);
            return outputs_2;
        }

        private List<float> sigmoid_mapper(List<List<float>> weights, List<float> inputs)
        {
            //if (weights.Count != inputs.Count)
            //    throw new ArgumentException("weights.Count and inputs.Count should be the same");

            List<float> tr = new List<float>();

            float x = 0;
            for (int j = 0; j < weights.Count; ++j)
            {
                for (int i = 0; i < inputs.Count; ++i)
                {
                    x += inputs[i] * weights[j][i];
                }
                var t = 1.0f / (1.0f + (float)System.Math.Exp(-x));
                tr.Add(t);
            }

            return tr;
        }

        internal float MSE(List<float> v, List<float> c)
        {
            var sum = 0D;
            for (int i = 0; i < v.Count; i++)
            {
                var a = v[i] - c[i];
                var a1 = Math.Pow(a, 2);
                sum = sum + a1;
            }
            return (float)sum;
        }
    }
}
