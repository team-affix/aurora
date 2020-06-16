using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AuroraCS.Sessions;
using AuroraCS.Modeling;
using AuroraCS.Maths;
using System.Runtime.Serialization;

namespace AuroraCS
{
    namespace Optimization
    {
        #region optimizers
        public class Optimizer
        {
            public double LearnRate;
            public Session Session;
            public virtual void UpdateGradient(object sender, double collectedGradient, double gradient, out double newCollectedGradient, out double newGradient)
            {
                newCollectedGradient = 0;
                newGradient = 0;
            }
            public virtual void UpdateParameter(object sender, double parameter, double collectedGradient, out double newParameter, out double newCollectedGradient)
            {
                newParameter = 0;
                newCollectedGradient = 0;
            }
        }
        public class BackPropagation : Optimizer
        {

        }
        public class SGD : BackPropagation
        {
            public SGD(double learnRate)
            {
                this.LearnRate = learnRate;
            }
            public override void UpdateGradient(object sender, double collectedGradient, double gradient, out double newCollectedGradient, out double newGradient)
            {
                newCollectedGradient = collectedGradient + gradient;
                newGradient = 0;
            }
            public override void UpdateParameter(object sender, double parameter, double collectedGradient, out double newParameter, out double newCollectedGradient)
            {
                newParameter = parameter - LearnRate * collectedGradient;
                newCollectedGradient = 0;
            }
        }
        public class Momentum : BackPropagation
        {
            public double Beta;
            public Momentum(double learnRate, double beta)
            {
                this.LearnRate = learnRate;
                this.Beta = beta;
            }
            public override void UpdateGradient(object sender, double collectedGradient, double gradient, out double newCollectedGradient, out double newGradient)
            {
                newCollectedGradient = Beta * collectedGradient + (1 - Beta) * gradient;
                newGradient = 0;
            }
            public override void UpdateParameter(object sender, double parameter, double collectedGradient, out double newParameter, out double newCollectedGradient)
            {
                newParameter = parameter - LearnRate * collectedGradient;
                newCollectedGradient = collectedGradient;
            }
        }
        public class Mutation : Optimizer
        {
            public double Beta;
            public List<MutationModelInfo> ModelInfos = new List<MutationModelInfo> { };
            int Index = 0;
            public _1D UpdateDistribution;
            public Mutation(double learnRate, double beta, _1D updateDistribution)
            {
                this.LearnRate = learnRate;
                this.Beta = beta;
                this.UpdateDistribution = updateDistribution;
            }
            public override void UpdateParameter(object sender, double parameter, double collectedGradient, out double newParameter, out double newCollectedGradient)
            {
                if (ModelInfos.Count < Index + 1)
                {
                    ModelInfos.Add(new MutationModelInfo());
                }
                MutationModelInfo m = ModelInfos[Index];
                m.RCV = UpdateDistribution[Session.Random.Next(0, UpdateDistribution.Count - 1)];
                double Update = (Beta * m.MovingAverage) + (1 - Beta) * m.RCV;
                newParameter = parameter + LearnRate * Update;
                newCollectedGradient = 0;
                Index++;
            }
            public void UpdateMovingAverages()
            {
                for (int i = 0; i < ModelInfos.Count; i++)
                {
                    MutationModelInfo m = ModelInfos[i];
                    m.MovingAverage = (Beta * m.MovingAverage) + (1 - Beta) * m.RCV;
                }
            }
            public void EndUpdate()
            {
                Index = 0;
            }
        }
        #endregion
        #region fluctuator
        [Serializable]
        public class Fluctuator
        {
        }
        #endregion
        #region Helpers
        public class MutationModelInfo
        {
            public double RCV;
            public double MovingAverage;
        }
        #endregion
    }
}
