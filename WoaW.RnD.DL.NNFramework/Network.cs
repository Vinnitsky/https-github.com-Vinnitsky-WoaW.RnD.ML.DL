using System;
using System.Collections.Generic;
using System.Linq;

namespace WoaW.RnD.DL.NNFramework
{
    public class Network
    {
        #region attributes
        private List<Dendrite> _dendrites;
        private Acson _acson;
        private List<Neuron> _neurons;
        private readonly int _outputPresision = 4;
        #endregion

        #region constructor
        public Network()
        {
            _dendrites = new List<Dendrite>();
            _acson = new Acson();
            _neurons = new List<Neuron>();
        }
        public Network(float learningRate, uint teachingNumber) : this()
        {
            LearningRate = learningRate;
            TeachingNumber = teachingNumber;
        }
        #endregion

        #region properties
        public IEnumerable<Dendrite> Dendrites
        {
            get { return _dendrites; }
            set { _dendrites = value as List<Dendrite>; }
        }
        public Acson Acson
        {
            get { return _acson; }
            set { _acson = value; }
        }
        public ICollection<Neuron> Neurons
        {
            get { return _neurons; }
            set { _neurons = value as List<Neuron>; }
        }
        public uint Epoch { get; set; }
        public uint TeachingNumber { get; set; }
        public float LearningRate { get; set; }
        public float Expected { get; set; }
        public Neuron this[string index]
        {
            get { return _neurons.SingleOrDefault(n => n.Id == index); }
            set
            {
                value.Id = index;
                _neurons.Add(value);
            }
        }
        #endregion

        public void Predict()
        {
            ForwardPropagation();

        }
        public void Teach()
        {
            for (int i = 0; i < TeachingNumber; i++)
            {
                ForwardPropagation();
                var forward = Slise();

                BackPropagation();
                var backward = Slise();

                Dump(Epoch, forward, backward);
                Epoch += 1;
            }
        }

        public void ForwardPropagation()
        {
            if (_neurons.Count == 0)
                throw new Exception("the network does not contain the neurons");

            //_neurons.ForEach(n => n.CalcuateValue());
            for (int i = 0; i < _neurons.Count; i++)
            {
                var neuron = _neurons[i];
                if (neuron.Type == NeuronType.Input)
                    continue;

                neuron.CalcuateValue();
            }
            Acson.Value = _neurons[_neurons.Count - 1].Acson.Value;
        }
        public void BackPropagation()
        {
            #region round
            {
                var neuron = _neurons[_neurons.Count - 1];
                neuron.Error = Acson.Value - Expected;
                if (neuron.Error == 0)
                    return;

                var r = neuron.Acson.Value;
                neuron.DeltaWeight = neuron.Error * neuron.DerivativeOfActivationFunction(r);

                foreach (var dendrite in neuron.Dendrites)
                {
                    dendrite.Value = dendrite.Value - dendrite.Acson.Value * neuron.DeltaWeight * LearningRate;
                }
            }
            #endregion

            for (int i = _neurons.Count - 2; i >= 0; i--)
            {
                var neuron = _neurons[i];
                if (neuron.Type == NeuronType.Input)
                    continue;

                var r = neuron.Acson.Value;

                foreach (var dendrite in neuron.Dendrites)
                {
                    neuron.Error = dendrite.Value * neuron.Acson.Dendrites.FirstOrDefault().Neuron.DeltaWeight;
                    neuron.DeltaWeight = neuron.Error * neuron.DerivativeOfActivationFunction(r);
                    dendrite.Value = dendrite.Value - dendrite.Acson.Value * neuron.DeltaWeight * LearningRate;
                }
                //prev_delta_weght = delta_weght; //оставляем глобальный или меняем всякий раз для новго слоя 
            }
        }

        public Tuple<Tuple<string, float>, List<float>>[] Slise()
        {
            var r = new Tuple<Tuple<string, float>, List<float>>[_neurons.Count];

            for (int j = 0; j < _neurons.Count; j++)
            {
                var neuron = _neurons[j];
                r[j] = new Tuple<Tuple<string, float>, List<float>>(new Tuple<string, float>(neuron.Id, neuron.Value), new List<float>());

                for (int i = 0; i < neuron.Dendrites.Count; i++)
                {
                    var dendrite = neuron.Dendrites[i];
                    r[j].Item2.Add(dendrite.Value);
                }
            }
            return r;
        }
        public void Dump(uint epoch,
            Tuple<Tuple<string, float>, List<float>>[] forward,
            Tuple<Tuple<string, float>, List<float>>[] backward = null)
        {
            var forward_str = "";

            foreach (var f in forward)
            {
                var t = "";
                for (int i = 0; i < f.Item2.Count; i++)
                {
                    t = t + string.Format(" [{0}]={1} ", i, System.Math.Round(f.Item2[i], _outputPresision));
                }
                forward_str = forward_str + string.Format("\t{0} v:{1} {2}" + System.Environment.NewLine, f.Item1.Item1, System.Math.Round(f.Item1.Item2, _outputPresision), t);
            }
            System.Diagnostics.Trace.Write("<-" + forward_str);

            if (backward == null)
                return;

            var backward_str = "";
            foreach (var f in backward)
            {
                var t = "";
                for (int i = 0; i < f.Item2.Count; i++)
                {
                    t = t + string.Format(" [{0}]={1} ", i, System.Math.Round(f.Item2[i], _outputPresision));
                }
                backward_str = backward_str + string.Format("\t{0} v:{1} {2}" + System.Environment.NewLine, f.Item1.Item1, System.Math.Round(f.Item1.Item2, _outputPresision), t);
            }
            System.Diagnostics.Trace.Write("<-" + backward_str);

            //System.Diagnostics.Debug.WriteLine(string.Format("->:{0}" + System.Environment.NewLine + "<-:{1}", forward_str, backward_str));

        }
    }
}
