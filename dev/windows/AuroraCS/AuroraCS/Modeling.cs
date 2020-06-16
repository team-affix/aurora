using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AuroraCS.Sessions;
using AuroraCS.Optimization;
using AuroraCS.Maths;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using AuroraCS.Extensions;

namespace AuroraCS
{
    namespace Modeling
    {
        [Serializable]
        public class Model
        {
            public dynamic Input;
            public dynamic InputGradient;
            public dynamic Output;
            public dynamic OutputGradient;
            [field: NonSerialized]
            public Session Session;
            public virtual void Initialize()
            {

            }
            public virtual void Forward()
            {

            }
            public virtual void Backward()
            {

            }
            public virtual void Update()
            {

            }
            public virtual void SendGradient(Model recipient)
            {

            }
            public virtual void SendState(Model recipient)
            {

            }
            public virtual void ModelWise(Action<Model> action)
            {
                action(this);
            }
        }
        [Serializable]
        public class Custom : Model
        {
            public Action<Custom> _Initialize;
            public Action<Custom> _Forward;
            public Action<Custom> _Backward;
            public Action<Custom> _Update;
            public Action<Custom, Model> _SendGradient;
            public Action<Custom, Model> _SendState;
            public Action<Custom, Action<Model>> _ModelWise;
            public override void Initialize()
            {
                if(_Initialize != null)
                {
                    _Initialize(this);
                }
                else
                {
                    base.Initialize();
                }
            }
            public override void Forward()
            {
                if (_Forward != null)
                {
                    _Forward(this);
                }
                else
                {
                    base.Forward();
                }
            }
            public override void Backward()
            {
                if (_Backward != null)
                {
                    _Backward(this);
                }
                else
                {
                    base.Backward();
                }
            }
            public override void Update()
            {
                if (_Update != null)
                {
                    _Update(this);
                }
                else
                {
                    base.Update();
                }
            }
            public override void SendGradient(Model recipient)
            {
                if (_SendGradient != null)
                {
                    _SendGradient(this, recipient);
                }
                else
                {
                    base.SendGradient(recipient);
                }
            }
            public override void SendState(Model recipient)
            {
                if (_SendState != null)
                {
                    _SendState(this, recipient);
                }
                else
                {
                    base.SendState(recipient);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                if (_ModelWise != null)
                {
                    _ModelWise(this, action);
                }
                else
                {
                    base.ModelWise(action);
                }
            }
        }
        [Serializable]
        public class Bias : Model
        {
            public double ParameterState;
            public double ParameterGradient;
            public double ParameterCollectedGradient;
            public Bias()
            {
                Input = 0;
                InputGradient = 0;
                Output = 0;
                OutputGradient = 0;

                this.ParameterState = 0;
                this.ParameterGradient = 0;
                this.ParameterCollectedGradient = 0;
            }
            public override void Initialize()
            {
                ParameterState = Session.InitializeParameter();
            }
            public override void Forward()
            {
                Output = (double)(object)Input + ParameterState;
            }
            public override void Backward()
            {
                ParameterGradient = (double)OutputGradient;
                InputGradient = OutputGradient;
            }
            public override void Update()
            {
                Session.Optimizer.UpdateParameter(this, ParameterState, ParameterCollectedGradient, out ParameterState, out ParameterCollectedGradient);
            }
            public override void SendGradient(Model recipient)
            {
                Bias b = (Bias)recipient;
                Session.Optimizer.UpdateGradient(this, b.ParameterCollectedGradient, ParameterGradient, out b.ParameterCollectedGradient, out ParameterGradient);
            }
            public override void SendState(Model recipient)
            {
                Bias b = (Bias)recipient;
                b.ParameterState = ParameterState;
            }
        }
        [Serializable]
        public class Activate : Model
        {
            public Activation Activation;
            public Activate(Activation activation)
            {
                Input = 0;
                InputGradient = 0;
                Output = 0;
                OutputGradient = 0;

                this.Activation = activation;

            }
            public override void Forward()
            {
                Output = Activation.Activate((double)Input);
            }
            public override void Backward()
            {
                InputGradient = (double)OutputGradient * Activation.Derivative((double)Output);
            }
        }
        [Serializable]
        public class Normalize : Model
        {
            public double ParameterState;
            public double ParameterGradient;
            public double ParameterCollectedGradient;
            public double Mu;
            public double Sigma;
            public double Beta;
            public Normalize(double beta, double sigma)
            {
                this.Beta = beta;
                this.Sigma = sigma;

                Input = 0;
                InputGradient = 0;
                Output = 0;
                OutputGradient = 0;

                this.ParameterState = 0;
                this.ParameterGradient = 0;
                this.ParameterCollectedGradient = 0;
            }
            public override void Initialize()
            {
                ParameterState = Session.InitializeParameter();
            }
            public override void Forward()
            {
                Output = 0.0001 * ParameterState * ((double)Input - Mu) / Sigma;
            }
            public override void Backward()
            {
                ParameterGradient = 0.0001 * (double)OutputGradient * (((double)Input - Mu) / Sigma);
                InputGradient = 0.0001 * (double)OutputGradient * ParameterState / Sigma;
            }
            public override void Update()
            {
                Session.Optimizer.UpdateParameter(this, ParameterState, ParameterCollectedGradient, out ParameterState, out ParameterCollectedGradient);

                Sigma = (Beta * Sigma) + (1 - Beta) * ((double)Input - Mu);
                Mu = (Beta * Mu) + (1 - Beta) * (double)Input;
            }
            public override void SendGradient(Model recipient)
            {
                Normalize n = (Normalize)recipient;
                Session.Optimizer.UpdateGradient(this, n.ParameterCollectedGradient, ParameterGradient, out n.ParameterCollectedGradient, out ParameterGradient);
                n.Input = (Beta * (double)n.Input) + (1 - Beta) * (double)Input;
            }
            public override void SendState(Model recipient)
            {
                Normalize n = (Normalize)recipient;
                n.ParameterState = ParameterState;
            }
        }
        [Serializable]
        public class Weight : Model
        {
            public double ParameterState;
            public double ParameterGradient;
            public double ParameterCollectedGradient;
            public Weight()
            {
                Input = 0;
                InputGradient = 0;
                Output = 0;
                OutputGradient = 0;

                this.ParameterState = 0;
                this.ParameterGradient = 0;
                this.ParameterCollectedGradient = 0;

            }
            public override void Initialize()
            {
                ParameterState = Session.InitializeParameter();
            }
            public override void Forward()
            {
                Output = (double)Input * ParameterState;
            }
            public override void Backward()
            {
                ParameterGradient = (double)OutputGradient * (double)Input;
                InputGradient = (double)OutputGradient * ParameterState;
            }
            public override void Update()
            {
                Session.Optimizer.UpdateParameter(this, ParameterState, ParameterCollectedGradient, out ParameterState, out ParameterCollectedGradient);
            }
            public override void SendGradient(Model recipient)
            {
                Weight w = (Weight)recipient;
                Session.Optimizer.UpdateGradient(this, w.ParameterCollectedGradient, ParameterGradient, out w.ParameterCollectedGradient, out ParameterGradient);
            }
            public override void SendState(Model recipient)
            {
                Weight w = (Weight)recipient;
                w.ParameterState = ParameterState;
            }
        }
        [Serializable]
        public class WeightSet : Model
        {
            public List<Weight> Weights = new List<Weight> { };
            public WeightSet(_1D parameterState)
            {
                parameterState.ForEach(x => { Weights.Add(new Weight()); });

                Input = 0;
                InputGradient = 0;
                Output = new _1D(Weights.Count);
                OutputGradient = new _1D(Weights.Count);

            }
            public override void Initialize()
            {
                Weights.ForEach(w => w.Initialize());
            }
            public override void Forward()
            {
                for (int i = 0; i < Weights.Count; i++)
                {
                    Weights[i].Input = Input;
                    Weights[i].Forward();
                    ((_1D)Output)[i] = (double)Weights[i].Output;
                }
            }
            public override void Backward()
            {
                double InputGradient = 0;
                for (int i = 0; i < Weights.Count; i++)
                {
                    Weights[i].OutputGradient = ((_1D)OutputGradient)[i];
                    Weights[i].Backward();
                    InputGradient += (double)Weights[i].InputGradient;
                }
                this.InputGradient = InputGradient;
            }
            public override void Update()
            {
                Weights.ForEach(w => { w.Update(); });
            }
            public override void SendGradient(Model recipient)
            {
                WeightSet w = (WeightSet)recipient;
                for (int i = 0; i < Weights.Count; i++)
                {
                    Weights[i].SendGradient(w.Weights[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                WeightSet w = (WeightSet)recipient;
                for (int i = 0; i < Weights.Count; i++)
                {
                    Weights[i].SendState(w.Weights[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                Weights.ForEach(w => w.ModelWise(action));
                base.ModelWise(action);
            }
        }
        [Serializable]
        public class WeightJunction : Model
        {
            public List<WeightSet> WeightSets = new List<WeightSet> { };
            public WeightJunction(_2D parameterState)
            {
                parameterState.ForEach(x => { WeightSets.Add(new WeightSet(x)); });

                Input = new _1D(parameterState.Rows);
                InputGradient = new _1D(parameterState.Rows);
                Output = new _1D(parameterState.Cols);
                OutputGradient = new _1D(parameterState.Cols);

            }
            public override void Initialize()
            {
                WeightSets.ForEach(w => w.Initialize());
            }
            public override void Forward()
            {
                _1D Output = new _1D(((_1D)this.Output).Count);
                for (int i = 0; i < WeightSets.Count; i++)
                {
                    WeightSets[i].Input = ((_1D)Input)[i];
                    WeightSets[i].Forward();
                    Output += (_1D)WeightSets[i].Output;
                }
                this.Output = Output;
            }
            public override void Backward()
            {
                _1D InputGradient = (_1D)this.InputGradient;
                for (int i = 0; i < WeightSets.Count; i++)
                {
                    WeightSets[i].OutputGradient = OutputGradient;
                    WeightSets[i].Backward();
                    InputGradient[i] = (double)WeightSets[i].InputGradient;
                }
                this.InputGradient = InputGradient;
            }
            public override void Update()
            {
                WeightSets.ForEach(w => { w.Update(); });
            }
            public override void SendGradient(Model recipient)
            {
                WeightJunction w = (WeightJunction)recipient;
                for (int i = 0; i < WeightSets.Count; i++)
                {
                    WeightSets[i].SendGradient(w.WeightSets[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                WeightJunction w = (WeightJunction)recipient;
                for (int i = 0; i < WeightSets.Count; i++)
                {
                    WeightSets[i].SendState(w.WeightSets[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                WeightSets.ForEach(w => w.ModelWise(action));
                base.ModelWise(action);
            }
        }
        [Serializable]
        public class Layer : Model
        {
            public List<Model> Models = new List<Model> { };
            public Layer(int a)
            {
                Input = new _1D(a);
                InputGradient = new _1D(a);
                Output = new _1D(a);
                OutputGradient = new _1D(a);
            }
            public Layer(int a, Func<Model> construct)
            {
                for (int i = 0; i < a; i++)
                {
                    Models.Add(construct());
                }

                Input = new _1D(a);
                InputGradient = new _1D(a);
                Output = new _1D(a);
                OutputGradient = new _1D(a);
            }
            public override void Initialize()
            {
                Models.ForEach(m => m.Initialize());
            }
            public override void Forward()
            {
                List<object> Output = (List<object>)this.Output;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].Input = ((List<object>)Input)[i];
                    Models[i].Forward();
                    Output[i] = Models[i].Output;
                }
                this.Output = Output;
            }
            public override void Backward()
            {
                List<object> InputGradient = (List<object>)this.InputGradient;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].OutputGradient = ((List<object>)OutputGradient)[i];
                    Models[i].Backward();
                    InputGradient[i] = Models[i].InputGradient;
                }
                this.InputGradient = InputGradient;
            }
            public override void Update()
            {
                Models.ForEach(m => { m.Update(); });
            }
            public override void SendGradient(Model recipient)
            {
                Layer l = (Layer)recipient;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].SendGradient(l.Models[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                Layer l = (Layer)recipient;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].SendState(l.Models[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                Models.ForEach(m => m.ModelWise(action));
                base.ModelWise(action);
            }
        }
        [Serializable]
        public class Sequential : Model
        {
            public List<Model> Models = new List<Model> { };
            public override void Initialize()
            {
                Models.ForEach(m => m.Initialize());
            }
            public override void Forward()
            {
                object CI = Input;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].Input = CI;
                    Models[i].Forward();
                    CI = Models[i].Output;
                }
                Output = CI;
            }
            public override void Backward()
            {
                object COG = OutputGradient;
                for (int i = Models.Count - 1; i >= 0; i--)
                {
                    Models[i].OutputGradient = COG;
                    Models[i].Backward();
                    COG = Models[i].InputGradient;
                }
                InputGradient = COG;
            }
            public override void Update()
            {
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].Update();
                }
            }
            public override void SendGradient(Model recipient)
            {
                Sequential s = (Sequential)recipient;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].SendGradient(s.Models[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                Sequential s = (Sequential)recipient;
                for (int i = 0; i < Models.Count; i++)
                {
                    Models[i].SendState(s.Models[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                Models.ForEach(m => m.ModelWise(action));
                base.ModelWise(action);
            }
            public void Add(Model model)
            {
                Models.Add(model);
            }
        }
        [Serializable]
        public class TNN : Sequential
        {
            public TNN(int[] units, Func<Model>[] constructNeurons)
            {
                this.Input = new _1D(units[0]);
                this.InputGradient = new _1D(units[0]);
                this.Output = new _1D(units[units.Length - 1]);
                this.OutputGradient = new _1D(units[units.Length - 1]);

                for (int i = 0; i < units.Length; i++)
                {
                    Add(new Layer(units[i], constructNeurons[i]));
                    if (units.Length > i + 1)
                    {
                        Add(new WeightJunction(new _2D(units[i], units[i + 1])));
                    }
                }
            }
        }
        [Serializable]
        public class Recurrent : Model
        {
            public List<Model> Unrolled = new List<Model> { };
            public List<Model> Prepared = new List<Model> { };
            public Model DefaultInstance;
            public int CarryIndex = -1;
            public Recurrent(Model defaultInstance)
            {
                this.Input = new _2D();
                this.InputGradient = new _2D();
                this.Output = new _2D();
                this.OutputGradient = new _2D();

                this.DefaultInstance = defaultInstance;
            }
            public override void Initialize()
            {
                DefaultInstance.Initialize();
            }
            public override void Forward()
            {
                IncrementForward(Unrolled.Count);
            }
            public override void Backward()
            {
                IncrementBackward(Unrolled.Count);
            }
            public override void Update()
            {
                DefaultInstance.Update();
            }
            public override void SendGradient(Model recipient)
            {
                Recurrent r = (Recurrent)recipient;
                for (int i = 0; i < Unrolled.Count; i++)
                {
                    Unrolled[i].SendGradient(r.Unrolled[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                Recurrent r = (Recurrent)recipient;
                for (int i = 0; i < Unrolled.Count; i++)
                {
                    Unrolled[i].SendState(r.Unrolled[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                Prepared.ForEach(p => p.ModelWise(action));
                DefaultInstance.ModelWise(action);
                base.ModelWise(action);
            }

            public void IncrementForward(int n)
            {
                _2D Output = (_2D)this.Output;
                for (int i = 0; i < n; i++)
                {
                    CarryIndex++;
                    Unrolled[CarryIndex].Input = ((_2D)Input)[CarryIndex];
                    Unrolled[CarryIndex].Forward();
                    Output[CarryIndex] = (_1D)Unrolled[CarryIndex].Output;
                }
                this.Output = Output;
            }
            public void IncrementBackward(int n)
            {
                _2D InputGradient = (_2D)this.InputGradient;
                for (int i = 0; i < n; i++)
                {
                    Unrolled[CarryIndex].OutputGradient = ((_2D)OutputGradient)[CarryIndex];
                    Unrolled[CarryIndex].Backward();
                    Unrolled[CarryIndex].SendGradient(DefaultInstance);
                    InputGradient[CarryIndex] = (_1D)Unrolled[CarryIndex].InputGradient;
                    CarryIndex--;
                }
                this.InputGradient = InputGradient;
            }
            public void Prep(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    Model model = DefaultInstance.Clone();
                    model.ModelWise(m => m.Session = Session);
                    Prepared.Add(model);
                }
            }
            public void Unroll(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    Next();
                }
            }
            public Model Next()
            {
                ((_2D)Input).Add(new _1D(((_1D)DefaultInstance.Input).Count));
                ((_2D)InputGradient).Add(new _1D(((_1D)DefaultInstance.Input).Count));
                ((_2D)Output).Add(new _1D(((_1D)DefaultInstance.Output).Count));
                ((_2D)OutputGradient).Add(new _1D(((_1D)DefaultInstance.Output).Count));

                Model Model = Prepared[Unrolled.Count];
                DefaultInstance.SendState(Model);
                Unrolled.Add(Model);
                return Model;
            }
            public void Clear()
            {
                Unrolled.Clear();
                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
            }
        }
        [Serializable]
        public class LSTM : Model
        {
            public List<LSTMTimeStep> Unrolled = new List<LSTMTimeStep> { };
            public List<LSTMTimeStep> Prepared = new List<LSTMTimeStep> { };
            public LSTMTimeStep DefaultInstance;
            public _1D hT;
            public _1D hTGrad;
            public _1D cT;
            public _1D cTGrad;
            public int CarryIndex = -1;
            static Func<Sequential> ConstructTanhNeuron = () => {
                Sequential Neuron = new Sequential();
                Neuron.Add(new Bias());
                Neuron.Add(new Activate(new Tanh()));
                return Neuron;
            };
            static Func<Sequential> ConstructSoftmaxNeuron = () => {
                Sequential Neuron = new Sequential();
                Neuron.Add(new Bias());
                Neuron.Add(new Activate(new Softmax()));
                return Neuron;
            };
            public LSTM(int inputUnits, int gateHiddenUnits, int outputUnits)
            {
                TNN ANetwork = new TNN(new int[] { inputUnits + outputUnits, gateHiddenUnits, outputUnits },
                new Func<Model>[] { ConstructTanhNeuron, ConstructTanhNeuron, ConstructSoftmaxNeuron });
                TNN BNetwork = new TNN(new int[] { inputUnits + outputUnits, gateHiddenUnits, outputUnits },
                new Func<Model>[] { ConstructTanhNeuron, ConstructTanhNeuron, ConstructSoftmaxNeuron });
                TNN CNetwork = new TNN(new int[] { inputUnits + outputUnits, gateHiddenUnits, outputUnits },
                new Func<Model>[] { ConstructTanhNeuron, ConstructTanhNeuron, ConstructTanhNeuron });
                TNN DNetwork = new TNN(new int[] { inputUnits + outputUnits, gateHiddenUnits, outputUnits },
                new Func<Model>[] { ConstructTanhNeuron, ConstructTanhNeuron, ConstructSoftmaxNeuron });

                hT = new _1D(outputUnits);
                hTGrad = new _1D(outputUnits);
                cT = new _1D(outputUnits);
                cTGrad = new _1D(outputUnits);

                DefaultInstance = new LSTMTimeStep(ANetwork, BNetwork, CNetwork, DNetwork, inputUnits, outputUnits, outputUnits);

                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
            }
            public LSTM(int inputUnits, int outputUnits, TNN ANetwork, TNN BNetwork, TNN CNetwork, TNN DNetwork)
            {
                hT = new _1D(outputUnits);
                hTGrad = new _1D(outputUnits);
                cT = new _1D(outputUnits);
                cTGrad = new _1D(outputUnits);

                DefaultInstance = new LSTMTimeStep(ANetwork, BNetwork, CNetwork, DNetwork, inputUnits, outputUnits, outputUnits);

                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
            }
            public override void Initialize()
            {
                DefaultInstance.Initialize();
            }
            public override void Forward()
            {
                IncrementForward(Unrolled.Count);
            }
            public override void Backward()
            {
                IncrementBackward(Unrolled.Count);
            }
            public override void Update()
            {
                DefaultInstance.Update();
            }
            public override void SendGradient(Model recipient)
            {
                LSTM l = (LSTM)recipient;
                DefaultInstance.SendGradient(l.DefaultInstance);
            }
            public override void SendState(Model recipient)
            {
                LSTM l = (LSTM)recipient;
                DefaultInstance.SendState(l.DefaultInstance);
            }
            public override void ModelWise(Action<Model> action)
            {
                DefaultInstance.ModelWise(action);
                Prepared.ForEach(p => p.ModelWise(action));
                base.ModelWise(action);
            }
            public void IncrementForward(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    CarryIndex++;
                    LSTMTimeStep timeStep = Unrolled[CarryIndex];
                    timeStep.hTIn = hT;
                    timeStep.cTIn = cT;
                    timeStep.Input = ((_2D)Input)[CarryIndex];
                    timeStep.Forward();
                    ((_2D)Output)[CarryIndex] = ((_1D)timeStep.Output);
                    hT = timeStep.hTOut;
                    cT = timeStep.cTOut;
                }
            }
            public void IncrementBackward(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    LSTMTimeStep timeStep = Unrolled[CarryIndex];
                    timeStep.hTOutGrad = hTGrad;
                    timeStep.cTOutGrad = cTGrad;
                    timeStep.OutputGradient = ((_2D)OutputGradient)[CarryIndex];
                    timeStep.Backward();
                    ((_2D)InputGradient)[CarryIndex] = ((_1D)timeStep.InputGradient);
                    hTGrad = timeStep.hTInGrad;
                    cTGrad = timeStep.cTInGrad;
                    timeStep.SendGradient(DefaultInstance);
                    CarryIndex--;
                }
            }

            public void Prep(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    LSTMTimeStep l = DefaultInstance.Clone();
                    l.ModelWise(m => m.Session = Session);
                    Prepared.Add(l);
                }
            }
            public void Unroll(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    Next();
                }
            }
            public LSTMTimeStep Next()
            {
                ((_2D)Input).Add(new _1D(((_1D)DefaultInstance.Input).Count));
                ((_2D)InputGradient).Add(new _1D(((_1D)DefaultInstance.Input).Count));
                ((_2D)Output).Add(new _1D(((_1D)DefaultInstance.Output).Count));
                ((_2D)OutputGradient).Add(new _1D(((_1D)DefaultInstance.Output).Count));

                LSTMTimeStep l = Prepared[Unrolled.Count];
                DefaultInstance.SendState(l);
                Unrolled.Add(l);
                return l;
            }
            public void Clear(bool stateful)
            {
                Unrolled.Clear();
                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
                if (!stateful)
                {
                    hT = new _1D(hT.Count);
                    hTGrad = new _1D(hT.Count);
                    cT = new _1D(cT.Count);
                    cTGrad = new _1D(cT.Count);
                }
            }
        }
        [Serializable]
        public class LSTMTimeStep : Model
        {
            public Sequential ANetwork;
            public Sequential BNetwork;
            public Sequential CNetwork;
            public Sequential DNetwork;
            public _1D hTIn;
            public _1D hTInGrad;
            public _1D hTOut;
            public _1D hTOutGrad;
            public _1D cTIn;
            public _1D cTInGrad;
            public _1D cTOut;
            public _1D cTOutGrad;
            public LSTMTimeStep(TNN aNetwork, TNN bNetwork, TNN cNetwork, TNN dNetwork, int xT, int hT, int cT)
            {
                this.ANetwork = aNetwork.Clone();
                this.BNetwork = bNetwork.Clone();
                this.CNetwork = cNetwork.Clone();
                this.DNetwork = dNetwork.Clone();

                this.Input = new _1D(xT);
                this.InputGradient = new _1D(xT);
                this.Output = new _1D(hT);
                this.OutputGradient = new _1D(hT);

                this.hTIn = new _1D(hT);
                this.hTInGrad = new _1D(hT);
                this.hTOut = new _1D(hT);
                this.hTOutGrad = new _1D(hT);
                this.cTIn = new _1D(cT);
                this.cTInGrad = new _1D(cT);
                this.cTOut = new _1D(cT);
                this.cTOutGrad = new _1D(cT);
            }
            public override void Initialize()
            {
                ANetwork.Initialize();
                BNetwork.Initialize();
                CNetwork.Initialize();
                DNetwork.Initialize();
            }
            public override void Forward()
            {
                _1D xT = ((_1D)Input).Concat(hTIn);
                ANetwork.Input = xT;
                BNetwork.Input = xT;
                CNetwork.Input = xT;
                DNetwork.Input = xT;
                ANetwork.Forward();
                BNetwork.Forward();
                CNetwork.Forward();
                DNetwork.Forward();
                cTOut = (cTIn ^ (_1D)ANetwork.Output) + ((_1D)BNetwork.Output ^ (_1D)CNetwork.Output);
                hTOut = (_1D)DNetwork.Output ^ cTOut;
                Output = hTOut;
            }
            public override void Backward()
            {
                _1D RealOutputGradient = hTOutGrad + (_1D)OutputGradient;
                DNetwork.OutputGradient = cTOut ^ RealOutputGradient;
                _1D RealcTGrad = cTOutGrad + ((_1D)DNetwork.Output ^ RealOutputGradient);
                CNetwork.OutputGradient = RealcTGrad ^ (_1D)BNetwork.Output;
                BNetwork.OutputGradient = RealcTGrad ^ (_1D)CNetwork.Output;
                ANetwork.OutputGradient = cTIn ^ RealcTGrad;
                cTInGrad = (_1D)ANetwork.Output ^ RealcTGrad;
                DNetwork.Backward();
                CNetwork.Backward();
                BNetwork.Backward();
                ANetwork.Backward();
                InputGradient = ((_1D)ANetwork.InputGradient).GetRange(0, ((_1D)Input).Count) + ((_1D)BNetwork.InputGradient).GetRange(0, ((_1D)Input).Count) + ((_1D)CNetwork.InputGradient).GetRange(0, ((_1D)Input).Count) + ((_1D)DNetwork.InputGradient).GetRange(0, ((_1D)Input).Count);
                hTInGrad = ((_1D)ANetwork.InputGradient).GetRange(((_1D)Input).Count, hTIn.Count) + ((_1D)BNetwork.InputGradient).GetRange(((_1D)Input).Count, hTIn.Count) + ((_1D)CNetwork.InputGradient).GetRange(((_1D)Input).Count, hTIn.Count) + ((_1D)DNetwork.InputGradient).GetRange(((_1D)Input).Count, hTIn.Count);
            }
            public override void Update()
            {
                ANetwork.Update();
                BNetwork.Update();
                CNetwork.Update();
                DNetwork.Update();
            }
            public override void SendGradient(Model recipient)
            {
                LSTMTimeStep l = (LSTMTimeStep)recipient;
                ANetwork.SendGradient(l.ANetwork);
                BNetwork.SendGradient(l.BNetwork);
                CNetwork.SendGradient(l.CNetwork);
                DNetwork.SendGradient(l.DNetwork);
            }
            public override void SendState(Model recipient)
            {
                LSTMTimeStep l = (LSTMTimeStep)recipient;
                ANetwork.SendState(l.ANetwork);
                BNetwork.SendState(l.BNetwork);
                CNetwork.SendState(l.CNetwork);
                DNetwork.SendState(l.DNetwork);
            }
            public override void ModelWise(Action<Model> action)
            {
                ANetwork.ModelWise(action);
                BNetwork.ModelWise(action);
                CNetwork.ModelWise(action);
                DNetwork.ModelWise(action);
                base.ModelWise(action);
            }
        }
        [Serializable]
        public class StackedLSTM : Model
        {
            List<LSTM> LSTMs = new List<LSTM> { };
            public int CarryIndex = -1;
            public StackedLSTM(List<LSTM> lSTMs)
            {
                this.LSTMs = lSTMs;

                this.Input = new _2D();
                this.InputGradient = new _2D();
                this.Output = new _2D();
                this.OutputGradient = new _2D();
            }
            public override void Initialize()
            {
                LSTMs.ForEach(l => { l.Initialize(); });
            }
            public override void Forward()
            {
                IncrementForward(LSTMs[0].Unrolled.Count);
            }
            public override void Backward()
            {
                IncrementBackward(LSTMs[0].Unrolled.Count);
            }
            public override void Update()
            {
                LSTMs.ForEach(l => { l.Update(); });
            }
            public override void SendGradient(Model recipient)
            {
                StackedLSTM s = (StackedLSTM)recipient;
                for (int i = 0; i < LSTMs.Count; i++)
                {
                    LSTMs[i].SendGradient(s.LSTMs[i]);
                }
            }
            public override void SendState(Model recipient)
            {
                StackedLSTM s = (StackedLSTM)recipient;
                for (int i = 0; i < LSTMs.Count; i++)
                {
                    LSTMs[i].SendState(s.LSTMs[i]);
                }
            }
            public override void ModelWise(Action<Model> action)
            {
                LSTMs.ForEach(l => l.ModelWise(action));
                base.ModelWise(action);
            }

            public void Prep(int n)
            {
                LSTMs.ForEach(l => { l.Prep(n); });
            }
            public void Unroll(int n)
            {
                for (int i = 0; i < n; i++)
                {
                    Next();
                }
            }
            public void Next()
            {
                ((_2D)Input).Add(new _1D(((_1D)LSTMs[0].DefaultInstance.Input).Count));
                ((_2D)InputGradient).Add(new _1D(((_1D)LSTMs[0].DefaultInstance.Input).Count));
                ((_2D)Output).Add(new _1D(((_1D)LSTMs[0].DefaultInstance.Output).Count));
                ((_2D)OutputGradient).Add(new _1D(((_1D)LSTMs[0].DefaultInstance.Output).Count));

                LSTMs.ForEach(l => { l.Next(); });
            }
            public void Clear(bool stateless)
            {
                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();

                LSTMs.ForEach(l => { l.Clear(stateless); });
            }
            public void IncrementForward(int n)
            {
                _2D Output = (_2D)this.Output;
                for (int i = 0; i < n; i++)
                {
                    CarryIndex++;
                    _1D CI = ((_2D)Input)[CarryIndex];
                    for (int j = 0; j < LSTMs.Count; j++)
                    {
                        ((_2D)LSTMs[j].Input)[CarryIndex] = CI;
                        LSTMs[j].IncrementForward(1);
                        CI = ((_2D)LSTMs[j].Output)[CarryIndex];
                    }
                    Output[CarryIndex] = CI;
                }
                this.Output = Output;
            }
            public void IncrementBackward(int n)
            {
                _2D InputGradient = (_2D)this.InputGradient;
                for (int i = 0; i < n; i++)
                {
                    _1D CG = ((_2D)OutputGradient)[CarryIndex];
                    for (int j = LSTMs.Count - 1; j >= 0; j--)
                    {
                        ((_2D)LSTMs[j].OutputGradient)[CarryIndex] = CG;
                        LSTMs[j].IncrementBackward(1);
                        CG = ((_2D)LSTMs[j].InputGradient)[CarryIndex];
                    }
                    InputGradient[CarryIndex] = CG;
                    CarryIndex--;
                }
            }
        }
        [Serializable]
        public class AttentionLSTM : Model
        {
            public List<Recurrent> Unrolled = new List<Recurrent> { };
            public List<Recurrent> Prepared = new List<Recurrent> { };
            public Recurrent DefaultInstance;
            public LSTM LSTM;
            static Func<Sequential> ConstructTanhNeuron = () => {
                Sequential Neuron = new Sequential();
                Neuron.Add(new Bias());
                Neuron.Add(new Activate(new Tanh()));
                return Neuron;
            };
            static Func<Sequential> ConstructSoftmaxNeuron = () => {
                Sequential Neuron = new Sequential();
                Neuron.Add(new Bias());
                Neuron.Add(new Activate(new Softmax()));
                return Neuron;
            };
            public AttentionLSTM(LSTM lSTM, int attentionHiddenUnits)
            {
                this.LSTM = lSTM;
                TNN AttentionTNN = new TNN(new int[] { ((_1D)lSTM.DefaultInstance.Input).Count + lSTM.hT.Count, attentionHiddenUnits, ((_1D)lSTM.DefaultInstance.Input).Count },
                new Func<Model>[]{
                ConstructTanhNeuron,
                ConstructTanhNeuron,
                ConstructSoftmaxNeuron
                });
                this.DefaultInstance = new Recurrent(AttentionTNN);

                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
            }
            public override void Initialize()
            {
                DefaultInstance.Initialize();
                LSTM.Initialize();
            }
            public override void Forward()
            {
                for (int i = 0; i < Unrolled.Count; i++)
                {
                    Recurrent recurrent = Unrolled[i];
                    LSTMTimeStep timeStep = LSTM.Unrolled[i];
                    _2D AttentionInput = new _2D();
                    ((_2D)Input).ForEach(x => { AttentionInput.Add(x.Concat(LSTM.hT)); });
                    recurrent.Input = AttentionInput;
                    recurrent.Forward();
                    _2D AttentionOutput = (_2D)recurrent.Output;
                    _2D LimitedInput = AttentionOutput ^ (_2D)Input;
                    _1D DecoderInput = LimitedInput.SumDown();
                    ((_2D)LSTM.Input).Add(DecoderInput);
                    ((_2D)LSTM.Output).Add(new _1D());
                    LSTM.IncrementForward(1);
                    ((_2D)Output)[i] = (_1D)timeStep.Output;
                }
            }
            public override void Backward()
            {
                LSTM.OutputGradient = OutputGradient;
                for (int i = Unrolled.Count - 1; i >= 0; i--)
                {
                    LSTMTimeStep timeStep = LSTM.Unrolled[i];
                    Recurrent recurrent = Unrolled[i];
                    LSTM.IncrementBackward(1);
                    recurrent.OutputGradient = new _2D();
                    ((_2D)Input).ForEach(x => { ((_2D)recurrent.OutputGradient).Add(x ^ (_1D)timeStep.InputGradient); });
                    recurrent.Backward();
                    _2D inputAdditionalAttentionGrad = ((_1D)timeStep.InputGradient).Repeat(((_2D)recurrent.Output).Rows) ^ (_2D)recurrent.Output;
                    _2D inputAttentionGrad = ((_2D)recurrent.InputGradient).GetCols(0, ((_1D)LSTM.DefaultInstance.Input).Count).Flip() + inputAdditionalAttentionGrad;
                    _1D hTAttentionGrad = ((_2D)recurrent.InputGradient).GetCols(((_1D)LSTM.DefaultInstance.Input).Count, ((_1D)recurrent.DefaultInstance.Input).Count - ((_1D)LSTM.DefaultInstance.Input).Count).SumAcross();
                    InputGradient = (_2D)InputGradient + inputAttentionGrad;
                    LSTM.hTGrad += hTAttentionGrad;
                }
            }
            public override void Update()
            {
                LSTM.Update();
                DefaultInstance.Update();
            }
            public override void SendGradient(Model recipient)
            {
                AttentionLSTM a = (AttentionLSTM)recipient;
                DefaultInstance.SendGradient(a.DefaultInstance);
                LSTM.SendGradient(a.LSTM);
            }
            public override void SendState(Model recipient)
            {
                AttentionLSTM a = (AttentionLSTM)recipient;
                DefaultInstance.SendState(a.DefaultInstance);
                LSTM.SendState(a.LSTM);
            }
            public override void ModelWise(Action<Model> action)
            {
                DefaultInstance.ModelWise(action);
                LSTM.ModelWise(action);
                Prepared.ForEach(p => p.ModelWise(action));
                base.ModelWise(action);
            }
            public void Clear()
            {
                LSTM.Clear(false);
                Unrolled.ForEach(x => { x.Clear(); });
                Unrolled.Clear();
                Input = new _2D();
                InputGradient = new _2D();
                Output = new _2D();
                OutputGradient = new _2D();
            }
            public void Unroll(int inputs, int outputs)
            {
                Input = new _2D(inputs, ((_1D)LSTM.DefaultInstance.Input).Count);
                InputGradient = new _2D(inputs, ((_1D)LSTM.DefaultInstance.Input).Count);
                Output = new _2D(outputs, ((_1D)LSTM.DefaultInstance.Output).Count);
                OutputGradient = new _2D(outputs, ((_1D)LSTM.DefaultInstance.Output).Count);

                LSTM.Unroll(outputs);
                for (int i = 0; i < outputs; i++)
                {
                    Recurrent recurrent = Prepared[i];
                    recurrent.Unroll(inputs);
                    Unrolled.Add(recurrent);
                }
            }
            public void Prep(int inputs, int outputs)
            {
                LSTM.Prep(outputs);
                for (int i = 0; i < outputs; i++)
                {
                    Recurrent recurrent = DefaultInstance.Clone();
                    recurrent.ModelWise(m => m.Session = Session);
                    recurrent.Prep(inputs);
                    Prepared.Add(recurrent);
                }
            }
        }
    }

    namespace Extensions
    {
        public static class Converter
        {
            public static byte[] ToByte(this object item)
            {
                var binFormatter = new BinaryFormatter();
                var mStream = new MemoryStream();
                binFormatter.Serialize(mStream, item);

                return mStream.ToArray();
            }
            public static object ToObject(this byte[] data)
            {
                var mStream = new MemoryStream();
                var binFormatter = new BinaryFormatter();
                // Where 'objectBytes' is your byte array.
                mStream.Write(data, 0, data.Length);
                mStream.Position = 0;
                return binFormatter.Deserialize(mStream) as object;
            }
            public static T Clone<T>(this T item)
            {
                byte[] data = item.ToByte();
                T Result = (T)data.ToObject();
                return Result;
            }
        }

    }
}