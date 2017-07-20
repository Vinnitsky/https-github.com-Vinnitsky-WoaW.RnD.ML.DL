using System.Collections.Generic;

namespace WoaW.RnD.DL.NNFramework
{
    using System;
    using System.Linq;

    public enum NeuronType
    {
        Undefined = 0,
        Input = 1,
        Heidien = 2,
        Output = 3,
        Bias = 4
    }

    public class Neuron
    {
        private List<Dendrite> _dendrites;
        private float _value;

        public Neuron()
        {
            ActivationFunction = Functions.Activation;
            DerivativeOfActivationFunction = Functions.Derivative;

            _dendrites = new List<Dendrite>();
            Acson = new Acson(this);
        }
        public Neuron(float value, NeuronType type, string id = null) : this()
        {
            Id = id;
            Value = value;
            Type = type;
        }

        #region properties
        public NeuronType Type { get; private set; }
        public List<Dendrite> Dendrites
        {
            get { return _dendrites; }
            private set { _dendrites = value; }
        }
        public Acson Acson { get; set; }
        public float Error { get; set; }
        public float DeltaWeight { get; set; }
        public string Id { get; set; }
        public Func<float, float> ActivationFunction { get; set; }
        public Func<float, float> DerivativeOfActivationFunction { get; set; }
        public float Value
        {
            get
            {
                if (Type == NeuronType.Bias)
                    return 1;
                else
                    return _value;
            }
            set { _value = value; Acson.Value = value; }
        }
        #endregion

        /// <summary>
        /// the method calculates neuron value. 
        /// since one neuron can have many connections with another neurons, the neuron value should be calculated as 
        /// sigmoid(x) where x is a sum of multiplications the previous neuron value on dendrite weight
        /// e=sum(v*w)
        /// </summary>
        public void CalcuateValue()
        {
            var sum = 0F;
            foreach (var dendrite in Dendrites)
            {
                sum = sum + dendrite.Acson.Value * dendrite.Value;
            }

            Value = ActivationFunction(sum);
            Acson.Value = Value;
        }

        internal void CalculateError()
        {
            if (Acson == null)
                return;

            var error = 0F;
            for (int j = 0; j < Acson.Dendrites.Count; j++)
            {
                var dendrite = Acson.Dendrites[j];
                error += dendrite.Value * dendrite.Neuron.Error;
            }
            Error = error;
        }
    }
}
