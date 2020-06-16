using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AuroraCS.Optimization;

namespace AuroraCS
{
    namespace Sessions
    {
        [Serializable]
        public class Session
        {
            public Random Random;
            public Optimizer Optimizer;
            public Fluctuator LearnRateFluctuator;
            public Func<double> InitializeParameter;
            public Session(Random random, Optimizer optimizer, Fluctuator learnRateFluctuator, Func<double> InitializeParameter)
            {
                this.Random = random;
                this.Optimizer = optimizer;
                this.Optimizer.Session = this;
                this.LearnRateFluctuator = learnRateFluctuator;
                this.InitializeParameter = InitializeParameter;
            }
        }
    }

}